Private Pfizer releases
======================

This integration branch combines independently upstreamable units without
squashing their histories. ``ci/internal-release.toml`` records their exact
tips and common upstream base. Packaging-only changes belong on this branch,
not the topic branches.

Versions use a PEP 440 local suffix, initially ``0.35.0+pfizer.1``. Never
overwrite an uploaded version. Tag validated commits with
``pfizer/cyvcf2/0.35.0+pfizer.1``; never use ``v*`` tags, which trigger the
upstream public PyPI release workflow. The private index is explicit, so
ordinary dependency resolution does not silently use it.

Build and validate
------------------

This private release targets ordinary GIL-enabled CPython 3.12 and newer using
``cp312-abi3`` wheels, not free-threaded ``abi3t``. The C extension is compiled
with ``Py_LIMITED_API`` through CMake's ``USE_SABI`` support, not merely
renamed. The release gate covers CPython 3.12, 3.13, and 3.14 on both platforms.
Build constraints use NumPy 2.5.4 headers with
``NPY_TARGET_VERSION=NPY_2_0_API_VERSION`` and retain the runtime
``numpy>=2.0.0`` requirement. Test the exact minimum NumPy on Python 3.12 and compatible newer
NumPy wheels on the other interpreters; do not silently raise the runtime floor.

Initialize submodules recursively and use a clean committed checkout. Install
build tools into an isolated environment, not the global Python installation.
The build requires uv, Cython, NumPy, scikit-build-core, CMake, Ninja, autoconf,
automake, and HTSlib's compression/curl development libraries. macOS additionally
requires delocate; Linux requires auditwheel and patchelf. Use CMake 3.26 or
newer for ``USE_SABI``. ``ci/internal-build-constraints.txt`` pins the Python
build dependencies used by the release script.

Run ``bash ci/build-internal-release /absolute/path/to/python dist/RELEASE``.
The output directory must be new. This builds the source distribution first,
builds the wheel from that archive with bundled static HTSlib, repairs external
shared libraries, and records artifact checksums and source provenance.
Keep the build log, dependency versions, and container image digest alongside
these outputs. Reproducible source provenance does not promise byte-identical
wheels across different compilers or build dependency versions.

For portable macOS compression libraries, first run
``bash ci/build-internal-macos-deps dist/macos-deps`` with CMake and Ninja on
``PATH``. Set ``CPPFLAGS=-I$PWD/dist/macos-deps/prefix/include``,
``LDFLAGS=-L$PWD/dist/macos-deps/prefix/lib``,
``PKG_CONFIG_PATH=$PWD/dist/macos-deps/prefix/lib/pkgconfig``, and
``MACOSX_DEPLOYMENT_TARGET=11.0`` for the release build. Do not mix in newer
Homebrew compression libraries, which may require a higher OS version.

Build macOS Apple Silicon wheels natively. Build Linux x86_64 wheels locally
inside a pinned ``manylinux_2_28_x86_64`` container using
``--platform linux/amd64``; verify ``uname -m`` reports ``x86_64``. An arm64
Docker host alone does not establish that the wheel targets x86_64.
``ci/Dockerfile.internal-manylinux`` pins the base image and installs native
dependencies. Mount only the final sdist, ``ci/`` scripts, and an output folder
into the local container, then run ``ci/build-internal-linux``. No repository
source is uploaded to a remote builder.
The pinned image's GCC 14 treats an incompatible-pointer warning in NumPy
2.0.0's own ``dtype_api.h`` as an error under the Limited API; its system GCC 8
also rejects an incomplete-type expression in the old headers. Newer build
headers targeting the NumPy 2.0 API avoid these compiler issues without
disabling diagnostics, modifying headers, or raising the runtime floor.
If corporate HTTPS requires existing local public CA certificates, pass their
bundle as the optional BuildKit secret ``trusted_ca``. Mount that same public
bundle and set ``SSL_CERT_FILE`` for container package downloads. Never disable
TLS verification or place private keys or credentials into the build context.

Inspect wheel dependencies with ``delocate-listdeps`` or ``auditwheel show``.
Run ``abi3audit --strict`` on both repaired wheels.
No Homebrew paths or external HTSlib dependencies may remain. Install each
repaired wheel in a fresh environment outside this source tree, check its
version and all four APIs, and run the complete combined source test suite
against that installed wheel. Inspect the sdist for all Cython/helper sources,
bundled HTSlib/htscodecs, and the exact-unit manifest; no local environment,
credentials, or build outputs may be included.
``ci/validate-internal-wheel`` creates an isolated environment, imports the
installed ABI3 extension outside the checkout, and runs all supplied source
tests as a package without shadowing the installed extension.

Publish and verify
------------------

Use the existing approved local credentials through
``UV_PUBLISH_USERNAME`` and ``UV_PUBLISH_PASSWORD``. Do not write credentials
into the repository, command-line arguments, logs, or chat. Do not change
global credential configuration.

Check the private simple index for the exact version and filenames before
uploading. A collision is a hard stop, not permission to overwrite or skip
an existing artifact. Confirm Artifactory accepts the local-version policy.
Select only the validated sdist and platform wheels, never a broad directory
containing unrepaired or stale artifacts::

    uv publish --index csartifacts --trusted-publishing never \
        /absolute/path/to/validated/cyvcf2-*.whl \
        /absolute/path/to/validated/cyvcf2-*.tar.gz

Never omit ``--index csartifacts`` or fall back to PyPI. Download the uploaded
files from the private index, compare their SHA-256 values with
``SHA256SUMS``, and verify a fresh exact-version install from that index::

    uv pip install --python /absolute/path/to/fresh/venv/bin/python \
        --index csartifacts "cyvcf2==0.35.0+pfizer.1"

Report publication as incomplete until both platform wheels, server metadata,
downloaded hashes, and clean installs have been verified.
