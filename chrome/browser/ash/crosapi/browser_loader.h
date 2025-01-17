// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_ASH_CROSAPI_BROWSER_LOADER_H_
#define CHROME_BROWSER_ASH_CROSAPI_BROWSER_LOADER_H_

#include <memory>

#include "base/files/file_path.h"
#include "base/functional/callback.h"
#include "base/gtest_prod_util.h"
#include "base/memory/scoped_refptr.h"
#include "base/memory/weak_ptr.h"
#include "base/time/time.h"
#include "chrome/browser/ash/crosapi/browser_util.h"

namespace component_updater {
class CrOSComponentManager;
}  // namespace component_updater

namespace crosapi {

class LacrosSelectionLoader;
using browser_util::LacrosSelection;

// Manages download of the lacros-chrome binary.
// This class is a part of ash-chrome.
class BrowserLoader {
 public:
  // Constructor for production.
  explicit BrowserLoader(
      scoped_refptr<component_updater::CrOSComponentManager> manager);
  // Constructor for testing.
  explicit BrowserLoader(
      std::unique_ptr<LacrosSelectionLoader> rootfs_lacros_loader,
      std::unique_ptr<LacrosSelectionLoader> stateful_lacros_loader);

  BrowserLoader(const BrowserLoader&) = delete;
  BrowserLoader& operator=(const BrowserLoader&) = delete;

  virtual ~BrowserLoader();

  // Returns true if the browser loader will try to load stateful lacros-chrome
  // builds from the component manager. This may return false if the user
  // specifies the lacros-chrome binary on the command line or the user has
  // forced the lacros selection to rootfs.
  // If this returns false subsequent loads of lacros-chrome will never load
  // a newer lacros-chrome version and update checking can be skipped.
  static bool WillLoadStatefulComponentBuilds();

  // Starts to load lacros-chrome binary or the rootfs lacros-chrome binary.
  // |callback| is called on completion with the path to the lacros-chrome on
  // success, or an empty filepath on failure, and the loaded lacros selection
  // which is either 'rootfs' or 'stateful'.
  using LoadCompletionCallback = base::OnceCallback<
      void(const base::FilePath&, LacrosSelection, base::Version)>;
  virtual void Load(LoadCompletionCallback callback);

  // Starts to unload lacros-chrome binary.
  // Note that this triggers to remove the user directory for lacros-chrome.
  virtual void Unload();

 private:
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest,
                           OnLoadSelectionQuicklyChooseRootfs);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest,
                           OnLoadVersionSelectionNeitherIsAvailable);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest,
                           OnLoadVersionSelectionStatefulIsUnavailable);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest,
                           OnLoadVersionSelectionRootfsIsUnavailable);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest,
                           OnLoadVersionSelectionRootfsIsNewer);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest,
                           OnLoadVersionSelectionRootfsIsOlder);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest, OnLoadSelectionPolicyIsRootfs);
  FRIEND_TEST_ALL_PREFIXES(
      BrowserLoaderTest,
      OnLoadSelectionPolicyIsUserChoiceAndCommandLineIsRootfs);
  FRIEND_TEST_ALL_PREFIXES(
      BrowserLoaderTest,
      OnLoadSelectionPolicyIsUserChoiceAndCommandLineIsStateful);
  FRIEND_TEST_ALL_PREFIXES(BrowserLoaderTest, OnLoadLacrosSpecifiedBySwitch);

  // `load_stateful_lacros` specifies whether we should start the installation
  // of stateful lacros in the background.
  void SelectRootfsLacros(LoadCompletionCallback callback,
                          bool load_stateful_lacros = false);
  void SelectStatefulLacros(LoadCompletionCallback callback);

  // Called when stateful lacros version is calculated.
  // TODO(crbug.com/1429138): Make it pararell to load stateful and rootfs
  // lacros.
  void OnLoadStatefulLacros(LoadCompletionCallback callback,
                            base::Version stateful_lacros_version);

  // Called to determine which lacros to load based on version (rootfs vs
  // stateful).
  void OnLoadVersionSelection(LoadCompletionCallback callback,
                              base::Version stateful_lacros_version,
                              base::Version rootfs_lacros_version);

  // Called on the completion of loading.
  void OnLoadComplete(LoadCompletionCallback callback,
                      LacrosSelection selection,
                      base::Version version,
                      const base::FilePath& path);
  void FinishOnLoadComplete(LoadCompletionCallback callback,
                            const base::FilePath& path,
                            LacrosSelection selection,
                            base::Version version,
                            bool lacros_binary_exists);

  // Loader for rootfs lacros and stateful lacros.
  std::unique_ptr<LacrosSelectionLoader> rootfs_lacros_loader_;
  std::unique_ptr<LacrosSelectionLoader> stateful_lacros_loader_;

  // Time when the lacros component was loaded.
  base::TimeTicks lacros_start_load_time_;

  base::WeakPtrFactory<BrowserLoader> weak_factory_{this};
};

}  // namespace crosapi

#endif  // CHROME_BROWSER_ASH_CROSAPI_BROWSER_LOADER_H_
