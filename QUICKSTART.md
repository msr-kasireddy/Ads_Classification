# Quick Start (no command line needed)

This guide is for using the tool as an end user. You won't type any commands.

## 1. Open the dashboard

- **macOS:** double-click **`run_dashboard.command`**
  - If macOS says *"cannot be opened because it is from an unidentified developer"*:
    right-click the file → **Open** → **Open**. (Only needed the first time.)
- **Windows:** double-click **`run_dashboard.bat`**

The first launch takes a few minutes (it sets itself up once). After that it
opens in a few seconds. Your web browser will open showing the dashboard.

> Leave the little black window open while you work — it's running the app.
> Close it when you're done.

## 2. Point it at your newspaper images

In the sidebar (**① Your pages**), set **Folder of newspaper images** to your
folder, for example:

```
/Users/reddy/KR/Sakshi/Sakshi-AI-Powered-Ads-Intelligence/apps/api/data/scraped_images
```

The pages appear. Use **⬅️ Prev / Next ➡️** (or the page number) to browse.

## 3. See what the model found

Each red box is an advertisement the current model detected, with its size in
**cm²** listed on the right. The built-in detector ("heuristic") is just a
*starting point* — it suggests candidate boxes. You make them correct:

- **Draw a missing ad:** choose **✏️ Draw new ad**, then drag a box over it.
- **Fix a wrong box:** choose **✋ Move / resize / delete**, click the box, then
  drag its handles to resize, drag the middle to move, or press **Delete** to
  remove it.

When the boxes on a page are correct, click **💾 Save my corrections for this
page**. Do this for as many pages as you can (a few hundred is the goal).

## 4. Train your own model

Scroll to **③ Train your own model**:

1. Click **① Set up the AI trainer** (one-time download). Reload when it's done.
2. Click **② Train on my corrected pages**.

Training runs entirely on your computer (uses your GPU if you have one, else the
CPU — slower but still free). When it finishes, change the **Detector** in the
sidebar to **yolov8** to use your trained model. Each round of
correcting + training pushes accuracy higher.

---

**Everything is local and free** — no internet account, no cloud, no cost.
The more pages you correct and train on, the closer you get to your accuracy goal.
