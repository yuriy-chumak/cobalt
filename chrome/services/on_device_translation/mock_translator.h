// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_SERVICES_ON_DEVICE_TRANSLATION_MOCK_TRANSLATOR_H_
#define CHROME_SERVICES_ON_DEVICE_TRANSLATION_MOCK_TRANSLATOR_H_

#include "chrome/services/on_device_translation/public/mojom/translator.mojom.h"

namespace on_device_translation {

class MockTranslator : public mojom::Translator {
 public:
  MockTranslator() = default;

  ~MockTranslator() override;

  MockTranslator(const MockTranslator&) = delete;
  MockTranslator& operator=(const MockTranslator&) = delete;

  static bool CanTranslate(const std::string& source_lang,
                           const std::string& target_lang);

  // mojom::Translator overrides.
  void Translate(const std::string& input, TranslateCallback callback) override;
};

}  // namespace on_device_translation

#endif  // CHROME_SERVICES_ON_DEVICE_TRANSLATION_MOCK_TRANSLATOR_H_
