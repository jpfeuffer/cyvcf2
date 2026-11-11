import ctypes
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing
import os
from pathlib import Path
import runpy
import threading
from urllib.parse import unquote, urlsplit

import pytest

from cyvcf2 import VCF, build_index
import cyvcf2.cyvcf2 as native
from .test_reader import coordinate_vcf


def _serve_s3(objects, requests, ready, stop):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, status, body=b"", headers=None, length=None):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body) if length is None else length))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def error(self, status, code):
            self.respond(status, ("<Error><Code>%s</Code></Error>" % code).encode())

        def do_GET(self):
            key = unquote(urlsplit(self.path).path)
            requests.append((self.command, self.headers.get("Range"), bool(self.headers.get("Authorization"))))
            payload = objects.get(key)
            if payload is None:
                self.error(404, "NoSuchKey")
                return
            headers = {"Accept-Ranges": "bytes"}
            range_header = self.headers.get("Range")
            if range_header:
                begin, end = range_header.removeprefix("bytes=").split("-")
                start = int(begin)
                stop = min(len(payload), int(end) + 1 if end else len(payload))
                if start >= len(payload):
                    self.error(416, "InvalidRange")
                    return
                headers["Content-Range"] = "bytes %d-%d/%d" % (start, stop - 1, len(payload))
                self.respond(206, payload[start:stop], headers)
            else:
                self.respond(200, payload, headers)

        def do_HEAD(self):
            key = unquote(urlsplit(self.path).path)
            if key not in objects:
                self.error(404, "NoSuchKey")
                return
            self.respond(200, length=len(objects[key]))

        def do_PUT(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            key = unquote(urlsplit(self.path).path)
            conditional = self.headers.get("If-None-Match")
            requests.append((self.command, None, conditional))
            with lock:
                if conditional != "*":
                    self.error(400, "MissingConditionalWrite")
                elif key in objects:
                    self.error(412, "PreconditionFailed")
                else:
                    objects[key] = payload
                    etag = hashlib.md5(payload, usedforsecurity=False).hexdigest()
                    self.respond(200, headers={"ETag": '"%s"' % etag})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    ready.send(server.server_address)
    ready.close()
    stop.wait()
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def s3_server(monkeypatch, tmp_path):
    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    objects, requests = manager.dict(), manager.list()
    receive, ready = context.Pipe(duplex=False)
    stop = context.Event()
    process = context.Process(target=_serve_s3, args=(objects, requests, ready, stop))
    process.start()
    ready.close()
    host, port = receive.recv()
    receive.close()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("HTS_S3_HOST", "%s:%s" % (host, port))
    monkeypatch.setenv("HTS_S3_ADDRESS_STYLE", "path")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://%s:%s" % (host, port))
    monkeypatch.setenv("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    monkeypatch.setenv("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
    try:
        yield objects, requests
    finally:
        stop.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("S3 test server did not stop")
        manager.shutdown()


@pytest.mark.parametrize("workers,suffix", [(1, "csi"), (4, "csi"), (2, "tbi")])
@pytest.mark.parametrize("protocol", ["s3+http", "http"])
def test_native_s3_input(s3_server, coordinate_vcf, tmp_path, workers, suffix, protocol):
    if not ctypes.CDLL(native.__file__).hts_features() & (1 << 11):
        pytest.skip("HTSlib was built without S3 support")
    objects, requests = s3_server
    payload = coordinate_vcf.read_bytes()
    objects["/cyvcf2-test/input.vcf.gz"] = payload
    source = (
        "s3+http://cyvcf2-test/input.vcf.gz" if protocol == "s3+http"
        else os.environ["AWS_ENDPOINT_URL_S3"] + "/cyvcf2-test/input.vcf.gz"
    )
    local = tmp_path / ("remote." + suffix)
    assert build_index(source, local, workers=workers, threads=1) == str(local)
    with pytest.raises(FileExistsError):
        build_index(source, local, workers=workers, threads=1)
    reference = tmp_path / ("reference." + suffix)
    build_index(coordinate_vcf, reference)
    assert gzip.decompress(local.read_bytes()) == gzip.decompress(reference.read_bytes())
    with VCF(source, samples=[]) as reader:
        reader.set_index(str(local))
        assert reader.num_records == 800
        assert next(reader("1:150-150")).POS <= 150
    if workers > 1:
        assert any(method == "GET" and ranged for method, ranged, _ in requests)
    if protocol == "s3+http":
        assert all(signed for method, _, signed in requests if method == "GET")
    assert objects["/cyvcf2-test/input.vcf.gz"] == payload
    assert all(method != "PUT" for method, _, _ in requests)


def test_native_s3_requires_local_destination():
    with pytest.raises(ValueError, match="explicit local index_path"):
        build_index("s3://bucket/input.vcf.gz")
    with pytest.raises(ValueError, match="destination must be local"):
        build_index("s3://bucket/input.vcf.gz", "s3://bucket/input.csi")
    with pytest.raises(ValueError, match="without credentials"):
        build_index("s3://name:password@bucket/input.vcf.gz", "input.csi")


@pytest.mark.parametrize("workers", [1, 4])
def test_native_s3_missing_input(s3_server, tmp_path, workers):
    if not ctypes.CDLL(native.__file__).hts_features() & (1 << 11):
        pytest.skip("HTSlib was built without S3 support")
    destination = tmp_path / "missing.csi"
    with pytest.raises(OSError):
        build_index("s3+http://cyvcf2-test/missing.vcf.gz", destination, workers=workers)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cyvcf2-index-*"))


def test_upath_s3_adapter(s3_server, coordinate_vcf, tmp_path, monkeypatch):
    pytest.importorskip("upath")
    pytest.importorskip("s3fs")
    script = Path(os.environ.get(
        "CYVCF2_TEST_S3_SCRIPT",
        str(Path(__file__).resolve().parents[2] / "scripts" / "build_index_s3.py"),
    ))
    if not script.exists():
        pytest.skip("optional adapter script is not installed with the wheel")
    adapter = runpy.run_path(str(script))["build_s3_index"]
    objects, requests = s3_server
    objects["/cyvcf2-test/input.vcf.gz"] = coordinate_vcf.read_bytes()
    result = adapter(
        "s3+http://cyvcf2-test/input.vcf.gz", "s3://cyvcf2-test/output.csi", workers=4, threads=1,
    )
    assert result == "s3://cyvcf2-test/output.csi"
    index = objects["/cyvcf2-test/output.csi"]
    local_index = tmp_path / "downloaded.csi"
    local_index.write_bytes(index)
    with VCF(str(coordinate_vcf), samples=[]) as reader:
        reader.set_index(str(local_index))
        assert reader.num_records == 800
    with pytest.raises(OSError):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/output.csi")
    assert objects["/cyvcf2-test/output.csi"] == index

    original = adapter.__globals__["build_index"]

    def concurrent_index(*args, **kwargs):
        result = original(*args, **kwargs)
        objects["/cyvcf2-test/race.csi"] = b"concurrent destination"
        return result

    monkeypatch.setitem(adapter.__globals__, "build_index", concurrent_index)
    with pytest.raises(OSError):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/race.csi")
    assert objects["/cyvcf2-test/race.csi"] == b"concurrent destination"
    assert all(conditional == "*" for method, _, conditional in requests if method == "PUT")
