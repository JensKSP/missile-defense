<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Jens Köhler -->
<!-- Assisted-by: Claude Code (Anthropic) -->

# Wayland

The game runs natively on Wayland. Nothing needs setting, and nothing should be:
it takes the session it finds, which on current KDE and GNOME means Wayland and
on an X11 login means X11.

That sentence took a while to be true, and this page is why — because the reason
is a defect in Qt that is still open, and the code working around it is one line
that looks like it does nothing.

## The defect

`QVulkanWindow` cannot survive Qt's own window teardown under Wayland. It is a
destruction order. `QWindowPrivate::destroy()` does this:

```cpp
q->setVisible(false);                                  // 1
QPlatformSurfaceEvent e(SurfaceAboutToBeDestroyed);
QGuiApplication::sendEvent(q, &e);                     // 2
delete std::exchange(platformWindow, nullptr);         // 3
```

Step 1 reaches `QWaylandWindow::reset()`, which tears down the `wl_surface`.
Step 2 is where `QVulkanWindow` destroys the swapchain built on that surface —
and the driver reads memory step 1 has already freed. The process dies with
SIGSEGV as the window closes, after the last frame, which is the most confusing
possible moment for it to happen.

Qt's own source comments on the hazard, in `QVulkanWindow::event`:

> The swapchain must be destroyed before the surface as per spec. This is not
> ideal for us because the surface is managed by the QPlatformWindow which may be
> gone already when the unexpose comes, making the validation layer scream. The
> solution is to listen to the PlatformSurface events.

Listening to those events is exactly what does not work here, because on Wayland
the surface dies one step earlier than that comment assumes.

Seen from the other side, the validation layer reports
`VUID-vkDestroySurfaceKHR-surface-01266` and libwayland prints
`wl_buffer still attached` warnings.

**Upstream:** [QTBUG-123214](https://bugreports.qt.io/browse/QTBUG-123214),
reported 2024-03-12 against Qt 6.6.2. Still untriaged, no fix version, no
comments. `QWindowPrivate::destroy()` is byte-identical in 6.9, 6.10 and `dev`,
so no Qt release available today changes this.

**Not one vendor's bug.** It reproduces identically under lavapipe, a software
rasteriser sharing no code with the NVIDIA driver, and not at all under X11 on
either. The surface is destroyed too early regardless of who owns the swapchain.

## The workaround

`GameWindow::event` releases the Vulkan instance when the window receives
`Close`:

```cpp
bool GameWindow::event(QEvent* event) {
    if (event->type() == QEvent::Close && QGuiApplication::platformName() == "wayland") {
        setVulkanInstance(nullptr);
    }
    return QVulkanWindow::event(event);
}
```

`Close` is the right moment because every way out of the game reaches it — the
menu's EXIT, the compositor's close button, a window manager asking — and it is
delivered immediately before `destroy()`, so nothing renders after it.

### What it costs

**The `VkSurfaceKHR` is never destroyed.** The platform window destroys that
surface through the instance it can reach from the window, and a detached window
offers it none.

That leak is the *mechanism*, not a side effect. The surface destruction Qt would
have performed is exactly the invalid one — the `wl_surface` beneath it is
already gone — so declining to perform it is the whole of the fix. The validation
layer sees the leak at shutdown as `VUID-vkDestroyInstance-instance-00629`.

It is an honest trade: one leaked handle in a process that is exiting, instead of
a segfault in a process that is exiting.

**This is why the workaround is gated on the platform.** On xcb the surface is
destroyed correctly and there is nothing to work around, so detaching there would
trade a clean teardown for a leaked handle and buy nothing. That is not a
hypothetical — applying it unconditionally is what made CI's Vulkan gate report
00629 on X11, which is how the cost above came to be understood at all.

With the line in place, Qt still calls `releaseSwapChainResources()` **and**
`releaseResources()` and the process exits 0 — the same callback sequence the
working X11 path produces:

| | `releaseSwapChainResources` | `releaseResources` | exit |
|---|---|---|---|
| Wayland, unmodified | dies inside it | never reached | SIGSEGV |
| Wayland, with the line | ✅ | ✅ | 0 |
| X11 (reference) | ✅ | ✅ | 0 |

Valgrind reports no invalid read, write or free anywhere in the teardown — the
leaked surface handle is a leak, not a corruption.

**How well this is understood.** The effect is measured, not derived. Nothing in
`QVulkanWindowPrivate::releaseSwapChain()` reads the instance, so Qt's source
does not explain why detaching changes its outcome. What is not in doubt is the
size of the effect: a bare `QVulkanWindow` crashes 24 of 24 runs without the line
and survives 24 of 24 with it, on two Vulkan implementations sharing no code.
That is not a timing coincidence.

Six other placements were tried first and all six still crashed — a device-wait
before `close()`, a device-wait in `releaseSwapChainResources()`, `quit()`
instead of `close()`, a queued close, `hide()` first, and minimise-to-unexpose —
because the free happens *inside* step 1, which every one of those paths goes
through.

## How it is kept honest

A workaround with no test decays into folklore: nobody dares remove it and
nobody can say what it does. `app/tests/wayland_teardown.cpp` is a bare
`QVulkanWindow` containing no line of this project, run in two modes, and
`python/tests/e2e/test_wayland_teardown.py` asserts three separate things:

* **the cause still exists** — the unmodified witness still dies. When Qt is
  fixed, this fails and says to delete the workaround;
* **the effect is the workaround's** — the same witness, with only that one line
  added, survives. This is what separates a fix from a coincidence;
* **the game still applies it** — the shipped binary exits 0 on native Wayland.

All three skip where there is no compositor, which includes CI. They are meant to
run on a developer's desktop, and `poe check` runs them there.

The tests decline to conclude anything from an AddressSanitizer build: its
quarantine keeps the freed block mapped, so the stale read succeeds and the crash
turns into a pass. They prefer the release build for that reason.

## Screenshots are the exception

`poe shot` still asks for `QT_QPA_PLATFORM=xcb`, because a Wayland client cannot
have its window grabbed by another process. That is the only place in the tree
that pins a platform, and `tools/capture.py` states the requirement where it
belongs. `poe app` and the ML console impose nothing.

## If you want the old behaviour

`QT_QPA_PLATFORM=xcb missile-defense` still works and always did. It costs
tearing — NVIDIA implements no implicit sync under XWayland — which is why it is
no longer the default.
