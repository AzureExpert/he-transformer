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

#include "he_backend.hpp"
#include "he_public_key.hpp"
#include "he_secret_key.hpp"
#include "ngraph/runtime/backend.hpp"
#include "openssl/bio.h"
#include "openssl/err.h"
#include "openssl/ssl.h"
#include "seal/seal.h"

namespace ngraph {
namespace runtime {
namespace he {
class HEPlaintextBackend : public HEBackend {
 public:
  HEPlaintextBackend(const std::string& server, const int port);

  // Opens a port
  int open_listener(int port);

  // Returns wherther or not the user is root
  bool is_root();

  // Initializes Client context
  SSL_CTX* init_Client_ctx();

  void load_certificates(SSL_CTX* ctx, char* CertFile, char* KeyFile);

  // prints certificates
  void show_certificates(SSL* ssl);
  void servlet(SSL* ssl); /* Serve the connection -- threadable */

  void* public_key;

  void wait_for_messages();

  void compute();

  // Computes funcation using m_server and m_server_port
  bool remote_call(std::shared_ptr<Function> function,
                   const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                   const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

  std::string m_server;  // Server to use for untrusted computation
  int m_port;            // Port at which server is running

  std::shared_ptr<runtime::Tensor> create_batched_cipher_tensor(
      const element::Type& element_type, const Shape& shape) override;

  std::shared_ptr<runtime::Tensor> create_batched_plain_tensor(
      const element::Type& element_type, const Shape& shape) override;

  void encode(std::shared_ptr<runtime::he::HEPlaintext>& output,
              const void* input, const element::Type& element_type,
              size_t count = 1) const override;

  void decode(void* output, const runtime::he::HEPlaintext* input,
              const element::Type& element_type,
              size_t count = 1) const override;

  void encrypt(std::shared_ptr<runtime::he::HECiphertext>& output,
               const runtime::he::HEPlaintext& input) const override;

  void decrypt(std::shared_ptr<runtime::he::HEPlaintext>& output,
               const runtime::he::HECiphertext& input) const override;

  bool call(
      std::shared_ptr<Function> function,
      const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

  std::shared_ptr<runtime::he::HECiphertext> create_empty_ciphertext() const =
      0;

  std::shared_ptr<runtime::he::HEPlaintext> create_empty_plaintext() const = 0;

 private:
  std::unique_ptr<HEPublicKey> m_public_key;
  std::unique_ptr<HESecretKey> m_secret_key;
  std::unique_ptr<runtime::Backend>
      m_backend;  // Backend to defer operations to
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph