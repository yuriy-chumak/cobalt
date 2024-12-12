// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'chrome-untrusted://lens/lens_overlay_app.js';

import {BrowserProxyImpl} from 'chrome-untrusted://lens/browser_proxy.js';
import type {LensOverlayAppElement} from 'chrome-untrusted://lens/lens_overlay_app.js';
import {UserAction} from 'chrome-untrusted://lens/metrics_utils.js';
import {assertEquals, assertFalse, assertTrue} from 'chrome-untrusted://webui-test/chai_assert.js';
import type {MetricsTracker} from 'chrome-untrusted://webui-test/metrics_test_support.js';
import {fakeMetricsPrivate} from 'chrome-untrusted://webui-test/metrics_test_support.js';
import {waitAfterNextRender, waitBeforeNextRender} from 'chrome-untrusted://webui-test/polymer_test_util.js';

import {TestLensOverlayBrowserProxy} from './test_overlay_browser_proxy.js';

suite('OverlayFeedbackButton', () => {
  let testBrowserProxy: TestLensOverlayBrowserProxy;
  let lensOverlayElement: LensOverlayAppElement;
  let metrics: MetricsTracker;

  setup(() => {
    // Resetting the HTML needs to be the first thing we do in setup to
    // guarantee that any singleton instances don't change while any UI is still
    // attached to the DOM.
    document.body.innerHTML = window.trustedTypes!.emptyHTML;

    testBrowserProxy = new TestLensOverlayBrowserProxy();
    BrowserProxyImpl.setInstance(testBrowserProxy);

    lensOverlayElement = document.createElement('lens-overlay-app');
    document.body.appendChild(lensOverlayElement);
    metrics = fakeMetricsPrivate();
    return waitBeforeNextRender(lensOverlayElement);
  });

  test('verify clicking more options toggles more options menu', async () => {
    // Menu should be hidden at first
    const moreOptionsMenu =
        lensOverlayElement.shadowRoot!.querySelector<HTMLElement>(
            '#moreOptionsMenu');
    assertEquals(window.getComputedStyle(moreOptionsMenu!).display, 'none');
    lensOverlayElement.$.moreOptionsButton.click();
    await waitAfterNextRender(lensOverlayElement);
    assertEquals(window.getComputedStyle(moreOptionsMenu!).display, 'flex');
    lensOverlayElement.$.moreOptionsButton.click();
    await waitAfterNextRender(lensOverlayElement);
    assertEquals(window.getComputedStyle(moreOptionsMenu!).display, 'none');
  });

  test('verify clicking my activity calls browser proxy', async () => {
    lensOverlayElement.shadowRoot!.querySelector<HTMLElement>('#myActivity')!
        .dispatchEvent(new MouseEvent('click', {
          button: 1,
          altKey: false,
          ctrlKey: true,
          metaKey: false,
          shiftKey: true,
        }));
    const clickModifiers =
        await testBrowserProxy.handler.whenCalled('activityRequestedByOverlay');
    assertTrue(clickModifiers.middleButton);
    assertFalse(clickModifiers.altKey);
    assertTrue(clickModifiers.ctrlKey);
    assertFalse(clickModifiers.metaKey);
    assertTrue(clickModifiers.shiftKey);
    assertEquals(1, metrics.count('Lens.Overlay.Overlay.UserAction'));
    assertEquals(
        1,
        metrics.count(
            'Lens.Overlay.Overlay.UserAction', UserAction.MY_ACTIVITY));
  });

  test('verify clicking learn more calls browser proxy', async () => {
    lensOverlayElement.shadowRoot!.querySelector<HTMLElement>('#learnMore')!
        .dispatchEvent(new MouseEvent('click', {
          button: 1,
          altKey: false,
          ctrlKey: true,
          metaKey: false,
          shiftKey: true,
        }));
    const clickModifiers =
        await testBrowserProxy.handler.whenCalled('infoRequestedByOverlay');
    assertTrue(clickModifiers.middleButton);
    assertFalse(clickModifiers.altKey);
    assertTrue(clickModifiers.ctrlKey);
    assertFalse(clickModifiers.metaKey);
    assertTrue(clickModifiers.shiftKey);
    assertEquals(1, metrics.count('Lens.Overlay.Overlay.UserAction'));
    assertEquals(
        1,
        metrics.count(
            'Lens.Overlay.Overlay.UserAction', UserAction.LEARN_MORE));
  });

  test('verify clicking send feedback calls browser proxy', () => {
    lensOverlayElement.shadowRoot!.querySelector<HTMLElement>(
                                      '#sendFeedback')!.click();
    assertEquals(1, metrics.count('Lens.Overlay.Overlay.UserAction'));
    assertEquals(
        1,
        metrics.count(
            'Lens.Overlay.Overlay.UserAction', UserAction.SEND_FEEDBACK));
    return testBrowserProxy.handler.whenCalled('feedbackRequestedByOverlay');
  });
});
