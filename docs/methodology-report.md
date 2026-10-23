# Methodology Report: Classify Waste Products Using Transfer Learning

## 1. Project Goal

The project develops an automated image classification system. The system separates recyclable waste from organic waste. It uses transfer learning with a pre-trained VGG16 network. The goal is to improve the efficiency of waste sorting at EcoClean and to reduce contamination of recyclable materials.

## 2. Dataset

The system uses the Waste Classification Dataset (reduced version with 1200 images).  
The data is organised as follows:

- Training folder: `o-vs-r-split/train/` containing sub-folders `O` (organic) and `R` (recyclable)
- Test folder: `o-vs-r-split/test/` containing the same class sub-folders

Images are resized to 150 × 150 pixels. The training set is split with 20 % held out for validation.

## 3. Data Preparation and Augmentation

Three ImageDataGenerator objects are created:

- **Training generator** – applies random width/height shifts and horizontal flips. Pixel values are rescaled to the range [0, 1].
- **Validation generator** – only rescales pixel values. No augmentation is applied.
- **Test generator** – only rescales pixel values. Shuffle is disabled so that predictions can be aligned with ground-truth labels.

Batch size is set to 32. The generators feed data continuously during training and evaluation.

## 4. Model Architecture – Feature Extraction Stage

1. Load the VGG16 network pre-trained on ImageNet.  
   - `include_top=False` removes the original classification head.  
   - Input shape is fixed at (150, 150, 3).

2. Flatten the last convolutional feature maps.

3. Freeze all layers of the base model so that their weights do not change.

4. Attach a new classification head:
   - Dense layer (512 units, ReLU)
   - Dropout (rate 0.3)
   - Dense layer (512 units, ReLU)
   - Dropout (rate 0.3)
   - Dense layer (1 unit, sigmoid) for binary classification

5. Compile the model:
   - Loss: binary cross-entropy
   - Optimiser: Adam with learning rate 1 × 10⁻⁵
   - Metric: accuracy

6. Train for up to 10 epochs with early stopping (patience = 4) and a learning-rate decay schedule. Model checkpoints save the best weights according to validation loss.

## 5. Fine-Tuning Stage

1. Re-load the VGG16 base model.

2. Unfreeze only the last convolutional block (`block5_conv3`) while keeping earlier layers frozen.

3. Rebuild the same classification head on top of the partially unfrozen base.

4. Compile with a slightly higher learning rate (RMSprop, 1 × 10⁻⁴).

5. Train again under the same early-stopping and checkpoint regime.

Fine-tuning allows the higher-level filters of VGG16 to adapt to the specific visual characteristics of waste images while preserving the generic low-level features learned from ImageNet.

## 6. Evaluation Procedure

- Load the best checkpoint from each training stage.
- Prepare a balanced test subset (50 organic + 50 recyclable images).
- Generate predictions and convert probabilities to class labels using a 0.5 threshold.
- Compute precision, recall, F1-score and support with scikit-learn’s classification report.
- Visualise individual test images together with the true label and the predicted label for both models.

## 7. Key Design Decisions and Rationale

| Decision | Reason |
|---|---|
| Transfer learning with VGG16 | Small dataset; ImageNet features transfer well to natural images |
| Freeze base then fine-tune only the last block | Prevents catastrophic forgetting while still allowing task-specific adaptation |
| Aggressive dropout (0.3) | Reduces over-fitting on a modest number of training samples |
| Early stopping + learning-rate decay | Stops training before validation performance degrades |
| Binary sigmoid output | Matches the two-class problem (organic vs recyclable) |

## 8. Expected Outcomes

- The feature-extraction model already achieves useful accuracy because of the strong ImageNet features.
- Fine-tuning typically raises accuracy further by adapting the top convolutional filters.
- Both models produce visual predictions that can be inspected for qualitative assessment.

## 9. Limitations and Future Improvements

- The reduced dataset (≈ 1200 images) limits absolute performance.
- Only one layer is unfrozen; unfreezing more layers with a very small learning rate could yield additional gains.
- Real-world deployment would require larger and more diverse data, possible multi-class expansion (plastic, paper, glass, metal, organic), and integration with a camera-based sorting system.

---

*This report summarises the complete methodology used to build, train, fine-tune and evaluate the waste classification models.*
