/*
 * CANN IPC Bindings for Python
 *
 * Provides fast CPython-native wrappers for the Ascend CANN IPC APIs:
 *   - aclrtIpcMemGetExportKey  (export device memory for IPC)
 *   - aclrtIpcMemImportByKey   (import device memory via IPC key)
 *   - aclrtIpcMemClose         (release IPC key)
 *
 * These bindings are an optional accelerator over the ctypes path in
 * npu_ipc_wrapper.py; when the C extension is importable it is preferred,
 * otherwise npu_ipc_wrapper falls back to ctypes transparently.
 *
 * Build (standalone):
 *   cd python/pegaflow/npu_ipc_bindings && python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>

/* =========================================================================
 * CANN API signatures (from acl/include/acl/acl_rt.h, CANN >= 8.5)
 * ========================================================================= */

typedef int (*aclrtIpcMemGetExportKey_fn)(void *devPtr,
                                          size_t size,
                                          char *key,
                                          size_t keyLen,
                                          unsigned long long flags);

typedef int (*aclrtIpcMemImportByKey_fn)(void **devPtr,
                                         const char *key,
                                         unsigned long long flags);

typedef int (*aclrtIpcMemClose_fn)(const char *key);


/* -------------------------------------------------------------------------
 * Shared library handle (loaded once, shared by all module functions)
 * ------------------------------------------------------------------------- */

static void *_libascendcl_handle = NULL;

static int
_load_libascendcl(void)
{
    const char *cann_home;
    char path[4096];
    int n;

    if (_libascendcl_handle != NULL) {
        return 0;  /* already loaded */
    }

    cann_home = getenv("ASCEND_HOME_PATH");
    if (cann_home != NULL) {
        n = snprintf(path, sizeof(path), "%s/lib64/libascendcl.so", cann_home);
        if (n >= (int)sizeof(path)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "ASCEND_HOME_PATH path too long");
            return -1;
        }
        _libascendcl_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (_libascendcl_handle != NULL) {
            return 0;
        }
    }

    /* Fall back to default search path */
    _libascendcl_handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (_libascendcl_handle == NULL) {
        const char *err = dlerror();
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot load libascendcl.so: %s. "
                     "Set ASCEND_HOME_PATH or add CANN runtime to LD_LIBRARY_PATH.",
                     err ? err : "unknown error");
        return -1;
    }
    return 0;
}

static void *
_get_fn(const char *name)
{
    void *fn = dlsym(_libascendcl_handle, name);
    if (fn == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot resolve %s from libascendcl.so", name);
        return NULL;
    }
    return fn;
}

/* =========================================================================
 * Python callables
 * ========================================================================= */

/*
 * export_key(dev_ptr: int, size: int) -> bytes
 *
 * Export a CANN IPC key for the given NPU memory region.
 * Returns the null-terminated key string as bytes.
 */
static PyObject *
npu_ipc_export_key(PyObject *self, PyObject *args)
{
    unsigned long long dev_ptr;
    unsigned long long size;
    char key_buf[256] = {0};
    aclrtIpcMemGetExportKey_fn fn;
    int ret;

    if (!PyArg_ParseTuple(args, "KK", &dev_ptr, &size)) {
        return NULL;
    }
    if (_load_libascendcl() != 0) {
        return NULL;
    }
    fn = (aclrtIpcMemGetExportKey_fn)_get_fn("aclrtIpcMemGetExportKey");
    if (fn == NULL) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    ret = fn((void *)dev_ptr, (size_t)size, key_buf, sizeof(key_buf), 0);
    Py_END_ALLOW_THREADS

    if (ret != 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "aclrtIpcMemGetExportKey failed with error %d "
                     "for dev_ptr=%p size=%llu",
                     ret, (void *)dev_ptr, size);
        return NULL;
    }
    return PyBytes_FromString(key_buf);
}

/*
 * import_key(key: bytes) -> int
 *
 * Import NPU memory via a CANN IPC key.
 * Returns the device virtual address as a Python int.
 */
static PyObject *
npu_ipc_import_key(PyObject *self, PyObject *args)
{
    const char *key;
    void *dev_ptr = NULL;
    aclrtIpcMemImportByKey_fn fn;
    int ret;

    if (!PyArg_ParseTuple(args, "y", &key)) {
        return NULL;
    }
    if (_load_libascendcl() != 0) {
        return NULL;
    }
    fn = (aclrtIpcMemImportByKey_fn)_get_fn("aclrtIpcMemImportByKey");
    if (fn == NULL) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    ret = fn(&dev_ptr, key, 0);
    Py_END_ALLOW_THREADS

    if (ret != 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "aclrtIpcMemImportByKey failed with error %d for key=%s",
                     ret, key);
        return NULL;
    }
    return PyLong_FromUnsignedLongLong((unsigned long long)dev_ptr);
}

/*
 * close_key(key: bytes) -> None
 *
 * Release a CANN IPC key. Idempotent; safe to call multuple times.
 */
static PyObject *
npu_ipc_close_key(PyObject *self, PyObject *args)
{
    const char *key;
    aclrtIpcMemClose_fn fn;
    int ret;

    if (!PyArg_ParseTuple(args, "y", &key)) {
        return NULL;
    }
    if (_load_libascendcl() != 0) {
        return NULL;
    }
    fn = (aclrtIpcMemClose_fn)_get_fn("aclrtIpcMemClose");
    if (fn == NULL) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    ret = fn(key);
    Py_END_ALLOW_THREADS

    if (ret != 0) {
        /*
         * close() failure is usually benign (double-close, already freed, etc.).
         * Log a warning rather than raising so clean-up code isn't disrupted.
         */
        PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                         "aclrtIpcMemClose returned %d for key=%s", ret, key);
    }
    Py_RETURN_NONE;
}


/* =========================================================================
 * Module definition
 * ========================================================================= */

static PyMethodDef _npu_ipc_methods[] = {
    {"export_key", npu_ipc_export_key, METH_VARARGS,
     "export_key(dev_ptr, size) -> bytes\n\n"
     "Export a CANN IPC key for NPU memory at dev_ptr with given size.\n"
     "Returns the key as a null-terminated bytes string."},
    {"import_key", npu_ipc_import_key, METH_VARARGS,
     "import_key(key) -> int\n\n"
     "Import NPU memory via a CANN IPC key.\n"
     "Returns the device virtual address as an integer."},
    {"close_key", npu_ipc_close_key, METH_VARARGS,
     "close_key(key) -> None\n\n"
     "Release a CANN IPC key."},
    {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef _npu_ipc_module = {
    PyModuleDef_HEAD_INIT,
    "_npu_ipc",
    "Low-level CANN IPC bindings for Ascend NPU memory sharing.",
    -1,
    _npu_ipc_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit__npu_ipc(void)
{
    return PyModule_Create(&_npu_ipc_module);
}