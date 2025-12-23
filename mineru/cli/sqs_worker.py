# Copyright (c) Opendatalab. All rights reserved.
import json
import os
import time
import asyncio
import base64
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import click

try:
    import boto3  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("boto3 is required to run sqs_worker") from exc

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    # Fallback: use a simple markdown converter
    try:
        import markdown2
        MARKDOWN_AVAILABLE = True
        markdown = markdown2  # Use markdown2 as fallback
    except ImportError:
        pass

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

from loguru import logger
from mineru.cli.common import aio_do_parse, read_fn


def _load_message_body(msg: Dict[str, Any]) -> Dict[str, Any]:
    body = msg.get("Body")
    if not body:
        return {}
    try:
        return json.loads(body)
    except Exception:
        # Allow plain text messages; treat as S3 key (advanced users can override)
        return {"input_key": body}


def _make_job_dir(base_dir: str, job_id: str) -> str:
    d = os.path.join(base_dir, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _upload_dir_to_s3(s3, local_dir: str, bucket: str, prefix: str):
    prefix = prefix.lstrip("/").rstrip("/")
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, start=local_dir)
            key = f"{prefix}/{rel}".lstrip("/")
            s3.upload_file(local_path, bucket, key)


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to convert image {image_path} to base64: {e}")
        return ""


def _replace_images_with_base64(markdown_text: str, image_dir: str) -> str:
    """Replace markdown image references with base64 data URIs"""
    pattern = r'\!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace(match):
        alt_text = match.group(1)
        relative_path = match.group(2)
        
        # Skip if already a data URI
        if relative_path.startswith('data:'):
            return match.group(0)
        
        # Try multiple path resolution strategies
        possible_paths = [
            os.path.join(image_dir, relative_path),  # Direct path
            os.path.join(image_dir, os.path.basename(relative_path)),  # Just filename
            relative_path,  # Absolute or relative to current dir
        ]
        
        full_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                full_path = path
                break
        
        # If not found, try with different extensions
        if not full_path:
            base_name = os.path.splitext(relative_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                for base_dir in [image_dir, os.path.dirname(relative_path) if relative_path else '.']:
                    test_path = os.path.join(base_dir, base_name + ext)
                    if os.path.exists(test_path) and os.path.isfile(test_path):
                        full_path = test_path
                        break
                if full_path:
                    break
        
        if full_path and os.path.exists(full_path):
            # Determine MIME type from extension
            ext = os.path.splitext(full_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            base64_image = _image_to_base64(full_path)
            if base64_image:
                logger.debug(f"Converted image to base64: {full_path}")
                return f'![{alt_text}](data:{mime_type};base64,{base64_image})'
            else:
                logger.warning(f"Failed to convert image to base64: {full_path}")
        else:
            logger.warning(f"Image not found: {relative_path} (searched in {image_dir})")
        
        # Return original if image not found
        return match.group(0)
    
    return re.sub(pattern, replace, markdown_text)


def _markdown_to_html(markdown_text: str) -> str:
    """Convert markdown text to HTML, preserving LaTeX math"""
    
    if not MARKDOWN_AVAILABLE:
        logger.warning("markdown library not available, using basic HTML conversion")
        # Very basic fallback - just escape HTML and preserve line breaks
        html = markdown_text.replace('\n', '<br>\n')
        html = html.replace('<', '&lt;').replace('>', '&gt;')
        return html
    
    try:
        # Protect LaTeX math before markdown conversion to prevent backslash stripping
        latex_blocks = []
        latex_inline = []
        
        # Protect block math \[...\] (can be multi-line)
        def protect_block_math(match):
            idx = len(latex_blocks)
            latex_blocks.append(match.group(0))
            return f"__LATEX_BLOCK_{idx}__"
        
        # Protect inline math \(...\)
        def protect_inline_math(match):
            idx = len(latex_inline)
            latex_inline.append(match.group(0))
            return f"__LATEX_INLINE_{idx}__"
        
        # Protect LaTeX before markdown processing
        protected_text = re.sub(r'\\\[[\s\S]*?\\\]', protect_block_math, markdown_text)
        protected_text = re.sub(r'\\\([^\\)]*?\\\)', protect_inline_math, protected_text)
        
        # Convert markdown to HTML
        if hasattr(markdown, 'markdown'):
            # Standard markdown library
            html = markdown.markdown(
                protected_text,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
        elif hasattr(markdown, 'convert'):
            # markdown2 library
            html = markdown.convert(protected_text, extras=['tables', 'fenced-code-blocks'])
        else:
            # Fallback
            html = protected_text.replace('\n', '<br>\n')
        
        # Restore LaTeX math (preserves all backslashes)
        for idx, latex in enumerate(latex_blocks):
            html = html.replace(f"__LATEX_BLOCK_{idx}__", latex)
        for idx, latex in enumerate(latex_inline):
            html = html.replace(f"__LATEX_INLINE_{idx}__", latex)
        
        # Post-process HTML to fix images and add block tagging
        html = _post_process_html(html)
        return html
    except Exception as e:
        logger.warning(f"Failed to convert markdown to HTML: {e}, using fallback")
        html = markdown_text.replace('\n', '<br>\n')
        return html


def _post_process_html(html: str) -> str:
    """Post-process HTML to fix images, math, and add block tagging similar to Marker"""
    if not BEAUTIFULSOUP_AVAILABLE:
        return html
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        block_counter = 0
        
        # Process images - ensure base64 data URIs are properly handled
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src.startswith('data:'):
                # Image is already base64, wrap in figure tag with data-block-id
                block_counter += 1
                block_id = f"image-{block_counter}"
                
                # Create figure wrapper
                figure = soup.new_tag('figure')
                figure['data-block-id'] = block_id
                figure['class'] = ['highlight-block']
                
                # Move img into figure
                img.extract()
                figure.append(img)
                
                # Insert figure where img was
                if img.parent:
                    img.parent.insert(img.parent.contents.index(img) if img in img.parent.contents else 0, figure)
                else:
                    # If no parent, wrap the img
                    img.wrap(figure)
            else:
                # Regular image, also wrap in figure
                block_counter += 1
                block_id = f"image-{block_counter}"
                figure = soup.new_tag('figure')
                figure['data-block-id'] = block_id
                figure['class'] = ['highlight-block']
                img.wrap(figure)
        
        # Add block IDs and highlight classes to other elements
        # Process paragraphs
        for p in soup.find_all('p'):
            if not p.get('data-block-id'):
                block_counter += 1
                p['data-block-id'] = f"block-{block_counter}"
                if 'highlight-block' not in p.get('class', []):
                    if p.get('class'):
                        p['class'].append('highlight-block')
                    else:
                        p['class'] = ['highlight-block']
        
        # Process tables
        for table in soup.find_all('table'):
            if not table.get('data-block-id'):
                block_counter += 1
                table['data-block-id'] = f"table-{block_counter}"
                wrapper = soup.new_tag('div')
                wrapper['class'] = ['highlight-block', 'highlight-table']
                wrapper['data-block-id'] = f"table-{block_counter}"
                table.wrap(wrapper)
        
        # Process headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if not heading.get('data-block-id'):
                block_counter += 1
                heading['data-block-id'] = f"heading-{block_counter}"
                if 'highlight-block' not in heading.get('class', []):
                    if heading.get('class'):
                        heading['class'].append('highlight-block')
                    else:
                        heading['class'] = ['highlight-block']
        
        # Process list items
        for li in soup.find_all('li'):
            if not li.get('data-block-id'):
                block_counter += 1
                li['data-block-id'] = f"list-item-{block_counter}"
        
        return str(soup)
    except Exception as e:
        logger.warning(f"Failed to post-process HTML: {e}")
        return html


def _wrap_html(body: str) -> str:
    """Wrap HTML body in a complete HTML document with styling similar to Marker"""
    return "\n".join([
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '    <meta charset="UTF-8">',
        "    <title>MinerU PDF Render</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; line-height: 1.6; }",
        "        .content { background: white; padding: 24px 32px; margin-bottom: 48px; box-shadow: 0 2px 6px rgba(0,0,0,0.12); }",
        "        img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; background: #fafafa; margin: 16px 0; }",
        "        figure { margin: 24px auto; text-align: center; }",
        "        figure img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; background: #fafafa; }",
        "        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }",
        "        pre { background: #f4f4f4; padding: 16px; border-radius: 4px; overflow-x: auto; }",
        "        table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
        "        table th, table td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }",
        "        table th { background-color: #f2f2f2; font-weight: bold; }",
        "        h1, h2, h3, h4, h5, h6 { margin-top: 24px; margin-bottom: 16px; }",
        "        p { margin: 12px 0; }",
        "        blockquote { border-left: 4px solid #ddd; padding-left: 16px; margin: 16px 0; color: #666; }",
        "        .highlight-block {",
        "            background: rgba(255, 235, 59, 0.35);",
        "            padding: 6px 8px;",
        "            border-radius: 6px;",
        "            margin: 6px 0;",
        "            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05);",
        "            cursor: pointer;",
        "        }",
        "        .highlight-block:hover {",
        "            background: rgba(255, 235, 59, 0.5);",
        "        }",
        "        .highlight-block figure {",
        "            background: white;",
        "        }",
        "        .highlight-table {",
        "            background: rgba(59, 130, 246, 0.2);",
        "            border: 1px solid rgba(59, 130, 246, 0.35);",
        "        }",
        "        [data-block-id] {",
        "            position: relative;",
        "        }",
        "    </style>",
        "    <script src='https://polyfill.io/v3/polyfill.min.js?features=es6'></script>",
        "    <script id='MathJax-script' async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>",
        "    <script>",
        "        window.MathJax = {",
        "            tex: {",
        "                inlineMath: [['\\\\(', '\\\\)']],",
        "                displayMath: [['\\\\[', '\\\\]']],",
        "                processEscapes: true,",
        "                processEnvironments: true",
        "            },",
        "            options: {",
        "                ignoreHtmlClass: '.*',",
        "                processHtmlClass: 'arithmatex'",
        "            }",
        "        };",
        "        // Enable block tagging functionality",
        "        document.addEventListener('DOMContentLoaded', function() {",
        "            document.querySelectorAll('[data-block-id]').forEach(function(block) {",
        "                block.addEventListener('click', function() {",
        "                    const blockId = this.getAttribute('data-block-id');",
        "                    console.log('Block ID:', blockId);",
        "                    // You can add custom tagging logic here",
        "                });",
        "            });",
        "        });",
        "    </script>",
        "</head>",
        "<body>",
        '    <div class="content">',
        body,
        "    </div>",
        "</body>",
        "</html>",
    ])


def render_markdown_to_html(md_path: Path, output_path: Path, image_dir: Optional[Path] = None) -> Path:
    """Render a MinerU markdown file to HTML"""
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    
    # Read markdown content
    md_content = md_path.read_text(encoding='utf-8')
    
    # Replace images with base64 if image directory is provided
    if image_dir and image_dir.exists():
        md_content = _replace_images_with_base64(md_content, str(image_dir))
    
    # Convert markdown to HTML
    html_body = _markdown_to_html(md_content)
    
    # Wrap in full HTML document
    full_html = _wrap_html(html_body)
    
    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html, encoding='utf-8')
    
    return output_path


async def _process_pdf_async(
    local_input: str,
    job_dir: str,
    backend: str,
    parse_method: str,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
    start_page_id: int,
    end_page_id: Optional[int],
    server_url: Optional[str],
    **kwargs,
):
    """Process PDF using MinerU's async API"""
    filename = os.path.basename(local_input) or "document"
    pdf_file_name = Path(filename).stem
    
    pdf_bytes = read_fn(local_input)
    
    await aio_do_parse(
        output_dir=job_dir,
        pdf_file_names=[pdf_file_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[lang],
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        **kwargs,
    )


@click.command(help="Poll SQS for S3 conversion jobs and process with MinerU.")
@click.option("--queue-url", required=True, help="SQS queue URL to poll")
@click.option("--region", default=None, help="AWS region (defaults to env/instance)")
@click.option("--input-bucket", default=None, help="Default S3 input bucket (if not in message)")
@click.option("--output-bucket", default=None, help="Default S3 output bucket (if not in message)")
@click.option(
    "--output-prefix",
    default="mineru-results",
    show_default=True,
    help="Default output prefix in output bucket",
)
@click.option(
    "--work-dir",
    default="/tmp/mineru_jobs",
    show_default=True,
    help="Local working directory on instance",
)
@click.option(
    "--wait-time",
    default=20,
    show_default=True,
    help="SQS long polling wait time (seconds, max 20)",
)
@click.option(
    "--visibility-timeout",
    default=3600,
    show_default=True,
    help="Visibility timeout in seconds (should exceed worst-case processing time)",
)
@click.option(
    "--delete-on-receive",
    is_flag=True,
    default=False,
    show_default=True,
    help="Delete the SQS message immediately after receiving/validating it (NOT recommended, but prevents retries).",
)
@click.option(
    "--backend",
    default="vlm-vllm-async-engine",
    show_default=True,
    help="MinerU backend to use (e.g., vlm-vllm-async-engine, pipeline, vlm-transformers)",
)
@click.option(
    "--parse-method",
    default="auto",
    type=click.Choice(["auto", "txt", "ocr"]),
    show_default=True,
    help="Parse method (only used with pipeline backend)",
)
@click.option(
    "--lang",
    default="ch",
    show_default=True,
    help="Language code (e.g., ch, en, korean, japan)",
)
@click.option(
    "--formula-enable/--no-formula-enable",
    default=True,
    show_default=True,
    help="Enable inline formula recognition. If disabled, inline formulas will not be detected or parsed.",
)
@click.option(
    "--table-enable/--no-table-enable",
    default=True,
    show_default=True,
    help="Enable table recognition. If disabled, tables will be shown as images.",
)
@click.option(
    "--server-url",
    default=None,
    help="Server URL for vlm-http-client backend",
)
@click.option(
    "--start-page-id",
    default=0,
    type=int,
    help="Starting page ID (0-indexed)",
)
@click.option(
    "--end-page-id",
    default=None,
    type=int,
    help="Ending page ID (0-indexed, None for all pages)",
)
def sqs_worker_cli(
    queue_url: str,
    region: Optional[str],
    input_bucket: Optional[str],
    output_bucket: Optional[str],
    output_prefix: str,
    work_dir: str,
    wait_time: int,
    visibility_timeout: int,
    delete_on_receive: bool,
    backend: str,
    parse_method: str,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: Optional[str],
    start_page_id: int,
    end_page_id: Optional[int],
):
    """
    Expected SQS message body JSON (minimum):

    {
      "input_bucket": "my-input-bucket",
      "input_key": "uploads/doc.pdf",
      "output_bucket": "my-output-bucket",
      "output_prefix": "mineru-results"
    }

    Optional message fields (override CLI defaults):
    - backend: MinerU backend to use
    - parse_method: Parse method (auto, txt, ocr)
    - lang: Language code
    - formula_enable: Enable inline formula recognition (true/false). If false, formulas won't be detected/parsed.
    - table_enable: Enable table recognition (true/false). If false, tables will be shown as images.
    - server_url: Server URL for vlm-http-client backend
    - start_page_id: Starting page ID
    - end_page_id: Ending page ID

    If output_bucket/prefix not provided in message, defaults come from CLI flags.
    """

    os.makedirs(work_dir, exist_ok=True)

    session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
    sqs = session.client("sqs")
    s3 = session.client("s3")

    logger.info("Starting MinerU SQS worker...")

    async def process_message(msg: Dict[str, Any]):
        receipt = msg["ReceiptHandle"]
        message_id = msg.get("MessageId", str(int(time.time())))

        body = _load_message_body(msg)
        in_bucket = body.get("input_bucket") or input_bucket
        in_key = body.get("input_key")
        out_bucket = body.get("output_bucket") or output_bucket
        out_prefix = body.get("output_prefix") or output_prefix

        # Override CLI defaults with message values if provided
        msg_backend = body.get("backend", backend)
        msg_parse_method = body.get("parse_method", parse_method)
        msg_lang = body.get("lang", lang)
        msg_formula_enable = body.get("formula_enable", formula_enable)
        msg_table_enable = body.get("table_enable", table_enable)
        msg_server_url = body.get("server_url", server_url)
        msg_start_page_id = body.get("start_page_id", start_page_id)
        msg_end_page_id = body.get("end_page_id", end_page_id)

        if not in_bucket or not in_key or not out_bucket:
            logger.error(f"Invalid message (missing buckets/keys): {body}")
            # Delete poison pill to avoid infinite retries
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            return

        job_id = body.get("job_id") or message_id
        job_dir = _make_job_dir(work_dir, job_id)
        filename = os.path.basename(in_key) or "document"
        local_input = os.path.join(job_dir, filename)

        try:
            if delete_on_receive:
                # NOTE: This trades reliability for simplicity: if the instance crashes mid-job,
                # the message is already gone and the job will not be retried.
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                receipt = None
                logger.info(f"[{job_id}] Deleted message on receive (no retries).")

            logger.info(f"[{job_id}] Downloading s3://{in_bucket}/{in_key}")
            s3.download_file(in_bucket, in_key, local_input)

            logger.info(f"[{job_id}] Processing with MinerU (backend: {msg_backend})...")
            await _process_pdf_async(
                local_input=local_input,
                job_dir=job_dir,
                backend=msg_backend,
                parse_method=msg_parse_method,
                lang=msg_lang,
                formula_enable=msg_formula_enable,
                table_enable=msg_table_enable,
                server_url=msg_server_url,
                start_page_id=msg_start_page_id,
                end_page_id=msg_end_page_id,
            )

            # Find and render markdown to HTML
            # MinerU outputs markdown files in subdirectories like: {pdf_file_name}/vlm/{pdf_file_name}.md
            # or {pdf_file_name}/auto/{pdf_file_name}.md (for pipeline backend)
            pdf_file_name = Path(filename).stem
            parse_method_dir = "vlm" if msg_backend.startswith("vlm-") else msg_parse_method
            md_path = Path(job_dir) / pdf_file_name / parse_method_dir / f"{pdf_file_name}.md"
            
            # Also try alternative paths
            if not md_path.exists():
                # Try without parse_method_dir
                md_path = Path(job_dir) / pdf_file_name / f"{pdf_file_name}.md"
            if not md_path.exists():
                # Try finding any .md file in the output directory
                md_files = list(Path(job_dir).rglob("*.md"))
                if md_files:
                    md_path = md_files[0]
            
            index_html_path = Path(job_dir) / "index.html"
            if md_path.exists():
                logger.info(f"[{job_id}] Rendering markdown to HTML...")
                # Try multiple possible image directory locations
                possible_image_dirs = [
                    md_path.parent / "images",  # Standard location
                    md_path.parent.parent / "images",  # One level up
                    Path(job_dir) / pdf_file_name / "images",  # At pdf_file_name level
                    Path(job_dir) / "images",  # At job level
                ]
                
                image_dir = None
                for img_dir in possible_image_dirs:
                    if img_dir.exists() and img_dir.is_dir():
                        image_dir = img_dir
                        logger.info(f"[{job_id}] Found images directory: {image_dir}")
                        break
                
                if not image_dir:
                    # Fallback to markdown directory
                    image_dir = md_path.parent
                    logger.warning(f"[{job_id}] Images directory not found, using markdown directory: {image_dir}")
                
                render_markdown_to_html(md_path, index_html_path, image_dir)
                logger.info(f"[{job_id}] HTML rendered to {index_html_path}")
            else:
                logger.warning(f"[{job_id}] Markdown file not found, skipping HTML rendering")

            # Upload everything in job_dir to S3 under output_prefix/job_id/
            out_root = f"{out_prefix.rstrip('/')}/{job_id}".strip("/")
            logger.info(f"[{job_id}] Uploading outputs to s3://{out_bucket}/{out_root}/")
            _upload_dir_to_s3(s3, job_dir, out_bucket, out_root)
            
            # Also upload stable names so the admin panel can fetch by job_id
            if md_path.exists():
                # Upload markdown file with stable name
                md_stable_path = Path(job_dir) / "document.md"
                if md_path != md_stable_path:
                    shutil.copy2(md_path, md_stable_path)
                s3.upload_file(str(md_stable_path), out_bucket, f"{out_root}/document.md")
            if index_html_path.exists():
                s3.upload_file(str(index_html_path), out_bucket, f"{out_root}/index.html")

            # Delete message only on success (unless already deleted on receive)
            if receipt is not None:
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            logger.info(f"[{job_id}] Done.")

        except Exception as exc:
            logger.exception(f"[{job_id}] ERROR: {exc}")
            # Do not delete message; it will retry after visibility timeout
            # You can send to DLQ by configuring redrive policy on the queue.

    async def worker_loop():
        while True:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=max(0, min(20, int(wait_time))),
                VisibilityTimeout=int(visibility_timeout),
            )

            msgs = resp.get("Messages", [])
            if not msgs:
                # No work
                continue

            await process_message(msgs[0])

    # Run the async worker loop
    asyncio.run(worker_loop())


if __name__ == "__main__":
    sqs_worker_cli()

