# Plan

1. Extract edits from lightroom
2. Filter to only predict the (global) edits I need
3. Clean up dataset. Remove unedited images, images with bad edits I don't want to train on
4. Extract features (dinov2 + other histogram based features + anything else I can think of)
5. Construct neural net predicting set of edit feature 
6. Construct train test split to make sure I'm not over-fitting on features I don't know
7. Train NN
8. Construct inference pipeline]
9. 