# ScabScout: 4-minute script (about 570 words, about 1 minute per speaker)

## HARSWITH (0:00 - 1:00)
**Slide 1, Cover (0:00)**: "Hi, we're Team Patriot. Our project is ScabScout. It looks at a photo of an apple leaf, outlines every diseased spot, and tells you which disease it is. On the test set it scores above 0.80 on precision, recall and F1, and it runs on a phone."

**Slide 2, What we did (0:15)**: "Here's our journey in four steps. We studied the data, we used a 2026 research paper, we tested eight workflow runs, each changing just one thing, and we deployed the winner in a phone app. Along the way, F1 went from 0.59 to 0.81."

**Slide 3, The problem (0:35)**: "So what makes this hard? Look at this chart: more than half the spots are smaller than 8 by 8 pixels. On a 256-pixel photo, that's tiny. So the real challenge isn't naming the disease, it's seeing the spot at all. And for a farmer, a missed spot is a missed infection, so recall mattered most to us. Devarsh?"

## DEVARSH (1:00 - 2:00)
**Slide 4, Research (1:00)**: "We found a 2026 study on exactly this problem. It showed that YOLO11, the model we use, struggles with small, blurry spots, which is what our data showed too. The authors added a new module to the model. We can't add parts in a no-code tool, so we did the next best thing: bigger images, so every spot gets more pixels. Outline quality went up about 30%. We also tried their training settings, and honestly, they didn't help us. They were tuned for a module we don't have."

**Slide 5, Experiments (1:25)**: "Here's every run. Run A is the guide as written: F1 0.59. In Run B we removed one preprocessing step, a colour shift, and F1 jumped to 0.77. That step recoloured the training photos but not the real ones, so the model was learning on leaves it would never see. Runs C and D used bigger images. E and F, the paper's settings, and G, a bigger model, didn't earn their place, so we dropped them. Because we changed one thing at a time, we know exactly what caused each gain. Thejan?"

## THEJAN (2:00 - 3:00)
**Slide 6, Final tuning (2:00)**: "Run H is where it clicked, and we didn't retrain anything. We studied the model's mistakes. 411 of its false spots were a second outline on a spot it had already found, so we remove duplicates. Its outlines were also too small, so we softened the mask edges. Then we gave each disease its own confidence cut-off, chosen on validation data. Step by step, recall climbed from 0.73 to 0.80, and every score crossed the line."

**Slide 7, Results (2:30)**: "Compared to the guided workflow, we find 68% more diseased spots: 9,828 of 12,263. And precision went up too, so we're not just guessing more."

**Slide 8, Workflow (2:45)**: "All of it runs in the ICICLE Workflow Studio: train, predict, then review. The predictions flow straight into the Smart Labeler. Yash?"

## YASH (3:00 - 4:00)
**Slide 9, Smart Labeler (3:00)**: "So we checked the model's work. Left: a black rot leaf, all 22 spots, nothing extra. Middle: its weak spot, apple scab, where it misses faint, blurry lesions. That's exactly why we pushed for bigger images. Right: it mistook a browned leaf edge for scab, and human review fixes that."

**Slide 10, Edge (3:20)**: "Why does this matter? Farmers often have no signal. So we kept the smallest YOLO11 model, 6 megabytes, and put it in an app that runs offline, right on the phone. Photo in, spots outlined, disease named."

**Slide 11, Takeaways (3:35)**: "What we learned: check the defaults, since one guide setting cost us the most. Tiny spots need pixels. How you read out a model matters as much as training it. And small is enough. Most of all, change one thing at a time and score honestly. Next, we'd build the read-out into the workflow and train longer. Thank you! We're happy to take questions."

## 1-minute Q&A
- Is 0.81 straight from the Studio? The model is Studio-trained; the Studio's own output is 0.84 / 0.73 / 0.78. Our post-processing script on the same model gets 0.82 / 0.81 / 0.81.
- Did you tune on the test set? The cut-offs were chosen on validation. We also report unseen leaves separately, since 74% of test photos are rotated training leaves.
- Why not a bigger model? YOLO11 Small was behind and 3x larger. For a phone, Nano wins.
- What's your edge device? A phone. The app runs the exact model offline in the browser.
- Why is scab hardest? Its lesions are faint and blurry with soft edges, so they're hard to see at 256 pixels.
