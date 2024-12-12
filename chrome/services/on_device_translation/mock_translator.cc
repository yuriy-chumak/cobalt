// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/services/on_device_translation/mock_translator.h"

#include "chrome/services/on_device_translation/public/mojom/translator.mojom.h"

namespace on_device_translation {

MockTranslator::~MockTranslator() = default;

// static
bool MockTranslator::CanTranslate(const std::string& source_lang,
                                  const std::string& target_lang) {
  return source_lang == target_lang;
}

void MockTranslator::Translate(const std::string& input,
                               TranslateCallback callback) {
  std::move(callback).Run(input);
}

}  // namespace on_device_translation
