import ctypes
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
import multiprocessing
import os
from pathlib import Path
import runpy
import threading
import tempfile
from urllib.parse import parse_qs, unquote, urlsplit
from xml.etree import ElementTree

import pytest

from cyvcf2 import VCF, build_index
import cyvcf2.cyvcf2 as native
from .test_reader import coordinate_vcf, _write_index_test_bgzf


def _serve_s3(objects, requests, ready, stop):
    lock = threading.Lock()
    uploads = {}
    upload_ids = count(1)

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
            query = parse_qs(urlsplit(self.path).query)
            conditional = self.headers.get("If-None-Match")
            requests.append((self.command, "part" if "uploadId" in query else None, conditional))
            with lock:
                if key == "/cyvcf2-test/fail-part.csi":
                    self.error(400, "InvalidRequest")
                elif "uploadId" in query:
                    upload = uploads[query["uploadId"][0]]
                    upload["parts"][int(query["partNumber"][0])] = payload
                    etag = hashlib.md5(payload, usedforsecurity=False).hexdigest()
                    self.respond(200, headers={"ETag": '"%s"' % etag})
                elif conditional != "*":
                    self.error(400, "MissingConditionalWrite")
                elif key in objects:
                    self.error(412, "PreconditionFailed")
                else:
                    objects[key] = payload
                    etag = hashlib.md5(payload, usedforsecurity=False).hexdigest()
                    self.respond(200, headers={"ETag": '"%s"' % etag})

        def do_POST(self):
            payload = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            key = unquote(urlsplit(self.path).path)
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            with lock:
                if "uploads" in query:
                    upload_id = str(next(upload_ids))
                    uploads[upload_id] = {"key": key, "parts": {}}
                    requests.append(("POST", "create", None))
                    self.respond(200, (
                        "<InitiateMultipartUploadResult><UploadId>%s</UploadId>"
                        "</InitiateMultipartUploadResult>" % upload_id
                    ).encode())
                    return
                upload_id = query["uploadId"][0]
                conditional = self.headers.get("If-None-Match")
                requests.append(("POST", "complete", conditional))
                if conditional != "*":
                    self.error(400, "MissingConditionalWrite")
                    return
                if key in objects:
                    self.error(412, "PreconditionFailed")
                    return
                upload = uploads[upload_id]
                parts = [
                    upload["parts"][int(part.findtext("{*}PartNumber"))]
                    for part in ElementTree.fromstring(payload).findall("{*}Part")
                ]
                if any(len(part) < 5 << 20 for part in parts[:-1]):
                    self.error(400, "EntityTooSmall")
                    return
                objects[key] = b"".join(parts)
                del uploads[upload_id]
                self.respond(200, b'<CompleteMultipartUploadResult><ETag>"completed"</ETag>'
                             b"</CompleteMultipartUploadResult>")

        def do_DELETE(self):
            query = parse_qs(urlsplit(self.path).query)
            with lock:
                uploads.pop(query["uploadId"][0], None)
            requests.append(("DELETE", "abort", None))
            self.respond(204)

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
@pytest.mark.parametrize("bounded", [False, True])
def test_native_s3_input(s3_server, coordinate_vcf, tmp_path, workers, suffix, protocol, bounded):
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
    events = []
    progress = events.append if bounded or workers > 1 else None
    assert build_index(
        source, local, workers=workers, threads=1, bounded=bounded, progress=progress,
    ) == str(local)
    if progress is not None:
        assert events[-1]["fraction"] == 1
        assert events[-1]["total_bytes"] == len(payload) * (2 if bounded else 1)
    with pytest.raises(FileExistsError):
        build_index(source, local, workers=workers, threads=1, bounded=bounded)
    reference = tmp_path / ("reference." + suffix)
    build_index(coordinate_vcf, reference)
    if not bounded:
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


@pytest.mark.parametrize("workers,suffix", [(1, "csi"), (4, "csi"), (2, "tbi")])
def test_upath_s3_adapter(s3_server, coordinate_vcf, tmp_path, monkeypatch, workers, suffix):
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

    def forbid_temporary_files(*args, **kwargs):
        pytest.fail("S3 indexing must not stage temporary files")

    for name in ("TemporaryDirectory", "TemporaryFile", "NamedTemporaryFile", "mkstemp"):
        monkeypatch.setattr(tempfile, name, forbid_temporary_files)

    destination = "s3://cyvcf2-test/output." + suffix
    result = adapter(
        "s3+http://cyvcf2-test/input.vcf.gz", destination, workers=workers, threads=1,
    )
    assert result == destination
    object_key = "/cyvcf2-test/output." + suffix
    index = objects[object_key]
    local_index = tmp_path / ("downloaded." + suffix)
    local_index.write_bytes(index)
    with VCF(str(coordinate_vcf), samples=[]) as reader:
        reader.set_index(str(local_index))
        assert reader.num_records == 800
    with pytest.raises(OSError):
        adapter(str(coordinate_vcf), destination)
    assert objects[object_key] == index

    original = native._build_index_to_fd

    def concurrent_index(*args, **kwargs):
        result = original(*args, **kwargs)
        objects["/cyvcf2-test/race.csi"] = b"concurrent destination"
        return result

    monkeypatch.setattr(native, "_build_index_to_fd", concurrent_index)
    with pytest.raises(OSError):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/race.csi")
    assert objects["/cyvcf2-test/race.csi"] == b"concurrent destination"
    assert all(conditional == "*" for method, action, conditional in requests if action == "complete")
    assert any(method == "DELETE" for method, _, _ in requests)


@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_streamed_index_region_equivalence(s3_server, tmp_path, monkeypatch, suffix):
    pytest.importorskip("upath")
    pytest.importorskip("s3fs")
    if not native._coordinate_workers_available:
        pytest.skip("coordinate workers require bundled HTSlib")
    from cyvcf2 import _index
    monkeypatch.setattr(_index, "_COMPRESSED_CHUNK_BYTES", 1024)
    path = tmp_path / "intervals.vcf.gz"
    header = (
        "##fileformat=VCFv4.3\n"
        "##contig=<ID=1,length=500000000>\n"
        "##contig=<ID=2,length=500000000>\n"
        "##INFO=<ID=END,Number=1,Type=Integer,Description=\"End\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    rows = []
    for chrom in ("1", "2"):
        for i in range(35001):
            position = i * 12000 + 1
            end = min(499999999, position + (2100000 if i % 101 == 0 else i % 17))
            rows.append(f"{chrom}\t{position}\tr{i}\tA\t<DEL>\t.\tPASS\tEND={end}\n")
    _write_index_test_bgzf(path, (header + "".join(rows)).encode())
    reference = tmp_path / ("reference." + suffix)
    build_index(path, reference)
    script = Path(os.environ.get(
        "CYVCF2_TEST_S3_SCRIPT",
        str(Path(__file__).resolve().parents[2] / "scripts" / "build_index_s3.py"),
    ))
    if not script.exists():
        pytest.skip("optional adapter script is not installed with the wheel")
    adapter = runpy.run_path(str(script))["build_s3_index"]
    objects, _ = s3_server
    adapter(str(path), "s3://cyvcf2-test/regions." + suffix, workers=4, threads=1)
    streamed = tmp_path / ("streamed." + suffix)
    streamed.write_bytes(objects["/cyvcf2-test/regions." + suffix])
    with VCF(str(path), samples=[]) as expected, VCF(str(path), samples=[]) as actual:
        expected.set_index(str(reference))
        actual.set_index(str(streamed))
        assert expected.num_records == actual.num_records == 70002
        positions = [1, 12000, 16384, 2100000, 100000000, 419999999, 500000000]
        positions += list(range(121000, 419000000, 5099431))
        for chrom in ("1", "2"):
            for position in positions:
                region = f"{chrom}:{position}-{position + 31}"
                assert [(v.POS, v.end, v.ID) for v in actual(region)] == [
                    (v.POS, v.end, v.ID) for v in expected(region)
                ]


def test_streamed_index_aborts_after_producer_failure(s3_server, coordinate_vcf, monkeypatch):
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

    def failed_producer(path, fd, **kwargs):
        data = b"x" * (6 << 20)
        with os.fdopen(os.dup(fd), "wb", buffering=0) as stream:
            stream.write(data)
        raise OSError("producer failed")

    monkeypatch.setattr(native, "_build_index_to_fd", failed_producer)
    with pytest.raises(OSError, match="producer failed"):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/failed.csi")
    assert "/cyvcf2-test/failed.csi" not in objects
    assert any(method == "DELETE" and action == "abort" for method, action, _ in requests)


def test_s3_progress_failure_does_not_publish(s3_server, coordinate_vcf):
    pytest.importorskip("upath")
    pytest.importorskip("s3fs")
    script = Path(os.environ.get(
        "CYVCF2_TEST_S3_SCRIPT",
        str(Path(__file__).resolve().parents[2] / "scripts" / "build_index_s3.py"),
    ))
    if not script.exists():
        pytest.skip("optional adapter script is not installed with the wheel")
    adapter = runpy.run_path(str(script))["build_s3_index"]
    objects, _ = s3_server

    def fail(event):
        if event["phase"] == "complete":
            raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/progress.csi", progress=fail)
    assert "/cyvcf2-test/progress.csi" not in objects


def test_streamed_index_rejects_input_changed_between_passes(s3_server, coordinate_vcf, monkeypatch):
    pytest.importorskip("upath")
    pytest.importorskip("s3fs")
    from cyvcf2 import _index
    script = Path(os.environ.get(
        "CYVCF2_TEST_S3_SCRIPT",
        str(Path(__file__).resolve().parents[2] / "scripts" / "build_index_s3.py"),
    ))
    if not script.exists():
        pytest.skip("optional adapter script is not installed with the wheel")
    adapter = runpy.run_path(str(script))["build_s3_index"]
    objects, _ = s3_server
    original = _index._run_coordinate_batches
    passes = 0

    def change_after_first_pass(*args):
        nonlocal passes
        original(*args)
        passes += 1
        if passes == 1:
            data = gzip.decompress(coordinate_vcf.read_bytes())
            _write_index_test_bgzf(coordinate_vcf, data.replace(b"1\t1\tr1\t", b"1\t2\tr1\t"))

    monkeypatch.setattr(_index, "_run_coordinate_batches", change_after_first_pass)
    with pytest.raises(OSError, match="changed between streaming passes"):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/changed.csi", workers=4)
    assert "/cyvcf2-test/changed.csi" not in objects


@pytest.mark.parametrize("suffix", ["csi", "tbi"])
def test_streamed_empty_vcf(s3_server, tmp_path, suffix):
    pytest.importorskip("upath")
    pytest.importorskip("s3fs")
    if not native._coordinate_workers_available:
        pytest.skip("coordinate workers require bundled HTSlib")
    script = Path(os.environ.get(
        "CYVCF2_TEST_S3_SCRIPT",
        str(Path(__file__).resolve().parents[2] / "scripts" / "build_index_s3.py"),
    ))
    if not script.exists():
        pytest.skip("optional adapter script is not installed with the wheel")
    path = tmp_path / "empty.vcf.gz"
    _write_index_test_bgzf(path, (
        "##fileformat=VCFv4.3\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    ).encode())
    adapter = runpy.run_path(str(script))["build_s3_index"]
    objects, _ = s3_server
    adapter(str(path), "s3://cyvcf2-test/empty." + suffix)
    index = tmp_path / ("empty." + suffix)
    index.write_bytes(objects["/cyvcf2-test/empty." + suffix])
    with VCF(str(path), samples=[]) as reader:
        reader.set_index(str(index))
        assert reader.num_records == 0


def test_streamed_multipart_conditional_completion(s3_server, coordinate_vcf, monkeypatch):
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
    data = b"x" * (11 << 20)

    def producer(path, fd, **kwargs):
        with os.fdopen(os.dup(fd), "wb", buffering=0) as stream:
            stream.write(data)

    monkeypatch.setattr(native, "_build_index_to_fd", producer)
    adapter(str(coordinate_vcf), "s3://cyvcf2-test/multipart.csi")
    assert objects["/cyvcf2-test/multipart.csi"] == data
    assert sum(method == "PUT" and action == "part" for method, action, _ in requests) >= 2
    assert ("POST", "complete", "*") in requests
    with pytest.raises(OSError):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/multipart.csi")
    assert objects["/cyvcf2-test/multipart.csi"] == data
    assert any(method == "DELETE" and action == "abort" for method, action, _ in requests)


def test_streamed_upload_failure_stops_producer(s3_server, coordinate_vcf, monkeypatch):
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

    def producer(path, fd, **kwargs):
        with os.fdopen(os.dup(fd), "wb", buffering=0) as stream:
            for _ in range(32):
                stream.write(b"x" * (1 << 20))

    monkeypatch.setattr(native, "_build_index_to_fd", producer)
    with pytest.raises(OSError):
        adapter(str(coordinate_vcf), "s3://cyvcf2-test/fail-part.csi")
    assert "/cyvcf2-test/fail-part.csi" not in objects
    assert any(method == "DELETE" and action == "abort" for method, action, _ in requests)
