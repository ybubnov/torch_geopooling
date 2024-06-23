#pragma once

#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>


inline void
TPHUtils_packDoubleArray(PyObject* tuple, size_t size, const double* numbers)
{
    for (size_t i = 0; i != size; ++i) {
        PyObject* number = PyFloat_FromDouble(numbers[i]);
        if (!number) {
            throw python_error();
        }
        PyTuple_SET_ITEM(tuple, i, number);
    }
}


inline PyObject*
TPHUtils_packDoubleArray(size_t size, const double* numbers)
{
    THPObjectPtr tuple(PyTuple_New(size));
    if (!tuple) {
        throw python_error();
    }
    TPHUtils_packDoubleArray(tuple.get(), size, numbers);
    return tuple.release();
}


namespace pybind11::detail {


template <> struct TORCH_PYTHON_API type_caster<c10::ArrayRef<double>> {
public:
    PYBIND11_TYPE_CASTER(c10::ArrayRef<double>, _("tuple[float, ...]"));

    bool
    load(handle src, bool)
    {
        PyObject* source = src.ptr();
        auto tuple = PyTuple_Check(source);

        if (tuple || PyList_Check(source)) {
            const auto size = tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
            v_value.resize(size);

            for (const auto idx : c10::irange(size)) {
                PyObject* obj
                    = (tuple ? PyTuple_GET_ITEM(source, idx) : PyList_GET_ITEM(source, idx));

                if (THPVariable_Check(obj)) {
                    v_value[idx] = THPVariable_Unpack(obj).item<double>();
                } else if (PyFloat_Check(obj)) {
                    v_value[idx] = THPUtils_unpackDouble(obj);
                } else {
                    return false;
                }
            }

            value = v_value;
            return true;
        }

        return false;
    }

    static handle
    cast(c10::ArrayRef<double> src, return_value_policy /* policy */, handle /* parent */)
    {
        return handle(TPHUtils_packDoubleArray(src.size(), src.data()));
    }

private:
    std::vector<double> v_value;
};


} // namespace pybind11::detail
