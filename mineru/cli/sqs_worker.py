# Copyright (c) Opendatalab. All rights reserved.
import json
import os
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import click

try:
    import boto3  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("boto3 is required to run sqs_worker") from exc

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
    """Upload all files from local_dir to S3, preserving subdirectory structure relative to local_dir"""
    prefix = prefix.lstrip("/").rstrip("/")
    local_dir = os.path.abspath(local_dir)  # Normalize path
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            # Get relative path from local_dir root
            rel = os.path.relpath(local_path, start=local_dir)
            # Normalize path separators for S3 (use forward slashes)
            rel = rel.replace(os.sep, "/")
            key = f"{prefix}/{rel}".lstrip("/")
            s3.upload_file(local_path, bucket, key)


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

            # Find MinerU output directory (structure: {pdf_file_name}/{parse_method}/)
            pdf_file_name = Path(filename).stem
            parse_method_dir = "vlm" if msg_backend.startswith("vlm-") else msg_parse_method
            mineru_output_dir = Path(job_dir) / pdf_file_name / parse_method_dir
            
            # Fallback: try to find any output directory if the expected one doesn't exist
            if not mineru_output_dir.exists():
                # Try alternative paths
                alt_paths = [
                    Path(job_dir) / pdf_file_name / msg_parse_method,
                    Path(job_dir) / pdf_file_name,
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        mineru_output_dir = alt_path
                        break
                else:
                    # If still not found, search for any subdirectory with .md files
                    md_files = list(Path(job_dir).rglob("*.md"))
                    if md_files:
                        mineru_output_dir = md_files[0].parent
                        logger.info(f"[{job_id}] Found MinerU output at: {mineru_output_dir}")
            
            if not mineru_output_dir.exists():
                logger.warning(f"[{job_id}] MinerU output directory not found, uploading from job_dir")
                mineru_output_dir = Path(job_dir)
            else:
                logger.info(f"[{job_id}] Uploading from MinerU output directory: {mineru_output_dir}")

            # Upload everything from MinerU output directory directly to output_prefix/job_id/
            # Files will be uploaded directly under job_id/ without the {pdf_file_name}/vlm/ nesting
            out_root = f"{out_prefix.rstrip('/')}/{job_id}".strip("/")
            logger.info(f"[{job_id}] Uploading raw MinerU outputs to s3://{out_bucket}/{out_root}/")
            logger.info(f"[{job_id}] Source directory: {mineru_output_dir.absolute()}")
            _upload_dir_to_s3(s3, str(mineru_output_dir.absolute()), out_bucket, out_root)

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

