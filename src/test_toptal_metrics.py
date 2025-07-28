#!/usr/bin/env python3
"""
Test script to verify Toptal metrics implementation
"""

from utils.misc import compute_toptal_metrics

def test_person_metrics():
    """Test person metrics (multi-word only)"""
    print("Testing Person Metrics (Multi-word only):")
    
    # Test case 1: "Joe Biden" should match "Joe Biden" but not ["Joe", "Biden"]
    y_true = [["Joe Biden", "John Smith"]]
    y_pred = [["Joe Biden", "John", "Smith"]]
    
    metrics = compute_toptal_metrics(y_true, y_pred, "person")
    print(f"True: {y_true[0]}")
    print(f"Pred: {y_pred[0]}")
    print(f"Multi-word Precision: {metrics['multi_word']['precision']:.3f}")
    print(f"Multi-word Recall: {metrics['multi_word']['recall']:.3f}")
    print(f"Multi-word F1: {metrics['multi_word']['f1']:.3f}")
    print()

def test_location_metrics():
    """Test location metrics (multi-word only)"""
    print("Testing Location Metrics (Multi-word only):")
    
    # Test case: "New York" should match "New York" but not ["New", "York"]
    y_true = [["New York", "Los Angeles"]]
    y_pred = [["New York", "New", "York"]]
    
    metrics = compute_toptal_metrics(y_true, y_pred, "location")
    print(f"True: {y_true[0]}")
    print(f"Pred: {y_pred[0]}")
    print(f"Multi-word Precision: {metrics['multi_word']['precision']:.3f}")
    print(f"Multi-word Recall: {metrics['multi_word']['recall']:.3f}")
    print(f"Multi-word F1: {metrics['multi_word']['f1']:.3f}")
    print()

def test_organization_metrics():
    """Test organization metrics (both multi-word and single-word)"""
    print("Testing Organization Metrics (Multi-word + Single-word):")
    
    # Test case: "Department of Agriculture" should match both:
    # 1. Multi-word: "Department of Agriculture" 
    # 2. Single-word: ["Department", "of", "Agriculture"]
    y_true = [["Department of Agriculture", "Microsoft"]]
    y_pred = [["Department of Agriculture", "Department", "Agriculture"]]
    
    metrics = compute_toptal_metrics(y_true, y_pred, "organization")
    print(f"True: {y_true[0]}")
    print(f"Pred: {y_pred[0]}")
    print("Multi-word metrics:")
    print(f"  Precision: {metrics['multi_word']['precision']:.3f}")
    print(f"  Recall: {metrics['multi_word']['recall']:.3f}")
    print(f"  F1: {metrics['multi_word']['f1']:.3f}")
    print("Single-word metrics:")
    print(f"  Precision: {metrics['single_word']['precision']:.3f}")
    print(f"  Recall: {metrics['single_word']['recall']:.3f}")
    print(f"  F1: {metrics['single_word']['f1']:.3f}")
    print()

def test_edge_cases():
    """Test edge cases"""
    print("Testing Edge Cases:")
    
    # Empty predictions
    y_true = [["Joe Biden", "New York"]]
    y_pred = [[]]
    
    metrics = compute_toptal_metrics(y_true, y_pred, "person")
    print(f"Empty predictions - Precision: {metrics['multi_word']['precision']:.3f}")
    print(f"Empty predictions - Recall: {metrics['multi_word']['recall']:.3f}")
    
    # Empty ground truth
    y_true = [[]]
    y_pred = [["Joe Biden"]]
    
    metrics = compute_toptal_metrics(y_true, y_pred, "person")
    print(f"Empty ground truth - Precision: {metrics['multi_word']['precision']:.3f}")
    print(f"Empty ground truth - Recall: {metrics['multi_word']['recall']:.3f}")
    print()

if __name__ == "__main__":
    test_person_metrics()
    test_location_metrics()
    test_organization_metrics()
    test_edge_cases()
    print("Toptal metrics implementation test completed!") 