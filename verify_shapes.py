import torch
import numpy as np
from model_architectures.dino_foundation_model import DinoFoundationModel
from model_architectures.object_detection_head import ObjectDetectionHead
from model_architectures.semantic_segmentation_head import SemanticSegmentationHead
from model_architectures.dav2_head import DepthAnythingV2Head

def verify_shapes():
    print("Starting Model Shape Verification...")

    # Configs
    encoder_size = "vits"
    batch_size = 1
    img_size = 518

    # 1. Foundation Model
    print(f"\n--- Verifying Foundation Model ({encoder_size}) ---")
    fm = DinoFoundationModel(encoder_size)
    fm.eval()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    with torch.no_grad():
        fm_output = fm(dummy_input)

    # The output of DinoFoundationModel.forward is a tuple of tuples (features, cls_token)
    # based on intermediate_layer_idx. Let's check the final one.
    final_features, final_cls = fm_output[-1]
    print(f"Foundation Model Output Shape: {final_features.shape}") # Expected: (B, N, C)

    # 2. Object Detection Head
    print(f"\n--- Verifying Object Detection Head ---")
    det_head = ObjectDetectionHead(encoder_size)
    det_head.eval()

    # The det_head expects a dict containing FM_OUTPUT_FEATURES
    from utils.naming_convention import FM_OUTPUT_FEATURES
    det_input = {FM_OUTPUT_FEATURES: final_features}

    with torch.no_grad():
        det_output = det_head.forward_annotated(det_input)

    print(f"Detection Head Output Labels Shape: {det_output['labels'].shape}")
    print(f"Detection Head Output Boxes Shape: {det_output['boxes'].shape}")

    # 3. Semantic Segmentation Head
    print(f"\n--- Verifying Semantic Segmentation Head ---")
    seg_head = SemanticSegmentationHead(encoder_size, dataset="road_safety")
    seg_head.eval()

    seg_input = {FM_OUTPUT_FEATURES: final_features}
    with torch.no_grad():
        seg_output = seg_head.forward_annotated(seg_input)

    from utils.naming_convention import MH_OUTPUT
    print(f"Segmentation Head Output Shape: {seg_output[MH_OUTPUT].shape}")

    # 4. Depth Head
    print(f"\n--- Verifying Depth Head ---")
    depth_head = DepthAnythingV2Head(encoder_size)
    depth_head.eval()

    # Depth head needs all intermediate features and cls tokens
    # We'll create dummy versions of those for the test
    depth_input = {}
    from utils.naming_convention import (
        FM_INTERMEDIATE_FEATURES_1, FM_INTERMEDIATE_CLS_TOKEN_1,
        FM_INTERMEDIATE_FEATURES_2, FM_INTERMEDIATE_CLS_TOKEN_2,
        FM_INTERMEDIATE_FEATURES_3, FM_INTERMEDIATE_CLS_TOKEN_3,
        FM_OUTPUT_FEATURES, FM_OUTPUT_CLS_TOKEN
    )

    # Mocking the tuple of 4 (features, cls_token) pairs
    depth_input[FM_INTERMEDIATE_FEATURES_1] = final_features.clone()
    depth_input[FM_INTERMEDIATE_CLS_TOKEN_1] = final_cls.clone()
    depth_input[FM_INTERMEDIATE_FEATURES_2] = final_features.clone()
    depth_input[FM_INTERMEDIATE_CLS_TOKEN_2] = final_cls.clone()
    depth_input[FM_INTERMEDIATE_FEATURES_3] = final_features.clone()
    depth_input[FM_INTERMEDIATE_CLS_TOKEN_3] = final_cls.clone()
    depth_input[FM_OUTPUT_FEATURES] = final_features.clone()
    depth_input[FM_OUTPUT_CLS_TOKEN] = final_cls.clone()

    with torch.no_grad():
        depth_output = depth_head.forward_annotated(depth_input)

    print(f"Depth Head Output Shape: {depth_output[MH_OUTPUT].shape}")

    print("\nAll shapes verified successfully!")

if __name__ == "__main__":
    verify_shapes()
