import re
import sys
import os
from pathlib import Path
from typing import Dict, List

from src.lamp_benchmark import create_metric, get_labels


class LAMPMetric:
    def __init__(self):
        pass

    def __call__(self, predictions) -> Dict[str, float]:
        """
        Compute LAMP metrics from predictions.
        
        Args:
            predictions: List of prediction dictionaries from the model
            
        Returns:
            Dict[str, float]: Computed metrics
        """
        # Extract predictions and ground truth
        predicted_answers = []
        ground_truth = []
        
        for sample in predictions:
            # Get predicted answer (string)
            pred = sample.get('predicted_answer', sample.get('prediction', ''))
            predicted_answers.append(pred)
            
            # Get ground truth - handle nested list structure [[answer]]
            gt = sample.get('answers', sample.get('ground_truth', ''))
            if isinstance(gt, list):
                # Flatten nested lists: [['[1]']] -> '[1]'
                while isinstance(gt, list) and len(gt) > 0:
                    gt = gt[0]
            ground_truth.append(str(gt))
        
        if not predicted_answers:
            return {"accuracy": 0.0}
        
        # Determine task from first sample if available
        task = "LaMP-1"  # Default
        if predictions and 'task' in predictions[0]:
            task_field = predictions[0]['task']
            # Handle task being a list: ['LaMP-1'] -> 'LaMP-1'
            if isinstance(task_field, list) and len(task_field) > 0:
                task = task_field[0]
            else:
                task = str(task_field)
        
        return self._compute_metrics(predicted_answers, ground_truth, task)

    def _compute_metrics(self, predicted_answers: List[str], ground_truth: List[str], task: str) -> Dict[str, float]:
        """Compute evaluation metrics for LAMP tasks"""
        
        # Clean and process predictions
        processed_predictions = []
        for pred in predicted_answers:
            processed_pred = self._process_prediction(pred, task)
            processed_predictions.append(processed_pred)
        
        # Compute metrics using LAMP's metric function
        try:
            lamp_metric = create_metric(task, average=True)
            results = lamp_metric(processed_predictions, ground_truth)
            
            # Flatten nested dictionaries and ensure all values are floats
            flattened_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            flattened_results[f"{key}_{sub_key}"] = float(sub_value)
                elif isinstance(value, (int, float)):
                    flattened_results[key] = float(value)
                elif isinstance(value, list):
                    # For lists, compute mean if all elements are numeric
                    if all(isinstance(x, (int, float)) for x in value):
                        flattened_results[f"{key}_mean"] = float(sum(value) / len(value))
            
            return flattened_results
            
        except Exception as e:
            print(f"Error computing LAMP metrics: {e}")
            # Return basic accuracy as fallback
            return self._compute_basic_accuracy(processed_predictions, ground_truth, task)
    
    def _process_prediction(self, prediction: str, task: str) -> str:
        """Process model predictions based on task type"""
        
        if task == "LaMP-1":
            # Citation identification - extract [1] or [2]
            match = re.search(r'\[([12])\]', prediction)
            if match:
                return f"[{match.group(1)}]"
            # Fallback: look for just the number
            match = re.search(r'\b([12])\b', prediction)
            if match:
                return f"[{match.group(1)}]"
            return "[1]"  # Default fallback
            
        elif task == "LaMP-2":
            # Movie tags - the ground truth is a single tag, so return the
            # first label mentioned in the output (joining all matches maps to
            # no valid label index downstream)
            prediction_lower = prediction.lower()
            try:
                labels = get_labels(task) if get_labels is not None else []
                hits = [
                    (prediction_lower.index(label.lower()), label)
                    for label in labels
                    if label.lower() in prediction_lower
                ]
                if hits:
                    return min(hits)[1]
            except Exception:
                pass
            return prediction.strip()
                
        elif task == "LaMP-3":
            # Rating prediction - extract 1-5 rating
            match = re.search(r'\b([1-5])\b', prediction)
            if match:
                return match.group(1)
            return "3"  # Default to middle rating
            
        else:
            # Generation tasks - return cleaned text
            return prediction.strip()
    
    def _compute_basic_accuracy(self, predictions: List[str], ground_truth: List[str], task: str) -> Dict[str, float]:
        """Compute basic accuracy as fallback metric"""
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truth):
            if task in ["LaMP-1", "LaMP-3"]:
                # Exact match for classification/regression
                if pred.strip() == gt.strip():
                    correct += 1
            elif task == "LaMP-2":
                # Partial match for tags
                pred_tags = set(pred.lower().split(", "))
                gt_tags = set(gt.lower().split(", "))
                if pred_tags & gt_tags:  # Any overlap
                    correct += 1
            else:
                # Basic string similarity for generation
                if pred.lower().strip() in gt.lower().strip() or gt.lower().strip() in pred.lower().strip():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}
