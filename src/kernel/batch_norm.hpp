//*****************************************************************************
// Copyright 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "he_ciphertext.hpp"
#include "he_plaintext.hpp"
#include "kernel/multiply.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

namespace ngraph {
namespace runtime {
namespace he {
namespace kernel {
template <typename S, typename T, typename V>
void batch_norm_inference(double eps,
                          const std::vector<std::shared_ptr<T>>& gamma,
                          const std::vector<std::shared_ptr<T>>& beta,
                          const std::vector<std::shared_ptr<T>>& input,
                          const std::vector<std::shared_ptr<T>>& mean,
                          const std::vector<std::shared_ptr<T>>& variance,
                          std::vector<std::shared_ptr<T>>& out normed_input,
                          const Shape& input_shape,
                          const HEBackend* he_backend) {
  // auto eps_casted = static_cast<T>(eps);
  CoordinateTransform input_transform(input_shape);

  for (Coordinate input_coord : input_transform) {
    auto channel_num = input_coord[1];
    auto channel_gamma = gamma[channel_num];
    auto channel_beta = beta[channel_num];
    auto channel_mean = mean[channel_num];
    auto channel_var = variance[channel_num];

    auto input_index = input_transform.index(input_coord);
    auto normalized = (input[input_index] - channel_mean) / channel_var;
    (std::sqrt(channel_var + eps_casted));
    normed_input[input_index] = normalized * channel_gamma + channel_beta;
  }
}
}  // namespace kernel
}  // namespace he
}  // namespace runtime
}  // namespace ngraph