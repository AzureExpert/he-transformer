/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <vector>

#include "he_backend.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/seal.h"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::multiply(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                                   const vector<shared_ptr<seal::Ciphertext>>& arg1,
                                   vector<shared_ptr<seal::Ciphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        he_backend.get()->get_evaluator()->multiply(*arg0[i], *arg1[i], *out[i]);
    }
}

void runtime::he::kernel::multiply(const shared_ptr<seal::Ciphertext>& arg0,
                                   const shared_ptr<seal::Ciphertext>& arg1,
                                   shared_ptr<seal::Ciphertext>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend)
{
    const vector<shared_ptr<seal::Ciphertext>> arg0vec = {arg0};
    const vector<shared_ptr<seal::Ciphertext>> arg1vec = {arg1};
    vector<shared_ptr<seal::Ciphertext>> outvec = {out};
    multiply(arg0vec, arg1vec, {outvec}, type, he_backend, 1);
}

void runtime::he::kernel::multiply(const vector<shared_ptr<seal::Ciphertext>>& arg0,
                                   const vector<shared_ptr<seal::Plaintext>>& arg1,
                                   vector<shared_ptr<seal::Ciphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            if (*arg1[i] == he_backend->get_plaintext_num().fl_1)
            {
                *out[i] = *arg0[i];
            }
            else if (*arg1[i] == he_backend->get_plaintext_num().fl_n1)
            {
                seal::Ciphertext c = *arg0[i];
                he_backend.get()->get_evaluator()->negate(c);
                *out[i] = c;
            }
            else
            {
                he_backend.get()->get_evaluator()->multiply_plain(*arg0[i], *arg1[i], *out[i]);
            }
        }
    }
    else if (type_name == "int64_t")
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            if (*arg1[i] == he_backend->get_plaintext_num().fl_1)
            {
                *out[i] = *arg0[i];
            }
            else if (*arg1[i] == he_backend->get_plaintext_num().fl_n1)
            {
                seal::Ciphertext c = *arg0[i];
                he_backend.get()->get_evaluator()->negate(c);
                *out[i] = c;
            }
            else
            {
                he_backend.get()->get_evaluator()->multiply_plain(*arg0[i], *arg1[i], *out[i]);
            }
        }
    }
    else if (type_name == "uint64_t")
    {
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            he_backend.get()->get_evaluator()->multiply_plain(*arg0[i], *arg1[i], *out[i]);
        }
    }
    else
    {
        throw ngraph_error("Multiply type not supported " + type_name);
    }
}

void runtime::he::kernel::multiply(const shared_ptr<seal::Ciphertext>& arg0,
                                   const shared_ptr<seal::Plaintext>& arg1,
                                   shared_ptr<seal::Ciphertext>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend)
{
    const vector<shared_ptr<seal::Ciphertext>> arg0vec = {arg0};
    const vector<shared_ptr<seal::Plaintext>> arg1vec = {arg1};
    vector<shared_ptr<seal::Ciphertext>> outvec = {out};
    multiply(arg0vec, arg1vec, {outvec}, type, he_backend, 1);
}

void runtime::he::kernel::multiply(const vector<shared_ptr<seal::Plaintext>>& arg0,
                                   const vector<shared_ptr<seal::Ciphertext>>& arg1,
                                   vector<shared_ptr<seal::Ciphertext>>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend,
                                   size_t count)
{
    multiply(arg1, arg0, out, type, he_backend, count);
}

void runtime::he::kernel::multiply(const shared_ptr<seal::Plaintext>& arg0,
                                   const shared_ptr<seal::Ciphertext>& arg1,
                                   shared_ptr<seal::Ciphertext>& out,
                                   const element::Type& type,
                                   shared_ptr<HEBackend> he_backend)
{
    multiply(arg1, arg0, out, type, he_backend);
}