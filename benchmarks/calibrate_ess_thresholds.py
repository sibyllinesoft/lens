#!/usr/bin/env python3
"""
ESS Threshold Calibration for Sanity Pyramid

Calibrate operation-specific ESS thresholds using ROC/PR curve optimization 
on a labeled validation set to maximize separation between "correctly answerable 
given context" vs "not yet answerable."
"""
import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_curve, precision_recall_curve, balanced_accuracy_score
import matplotlib.pyplot as plt

from sanity_pyramid import OperationType, SanityPyramid
from live_sanity_integration import LiveSanityIntegration, ESS_Thresholds
from code_search_rag_comprehensive import BenchmarkQuery

logger = logging.getLogger(__name__)


@dataclass
class LabeledQuery:
    """Query with ground truth answerability label."""
    query_id: str
    query: str
    operation: OperationType
    retrieved_chunks: List[Dict]
    gold_data: Dict
    ess_score: float
    is_answerable: bool  # Ground truth: can this query be correctly answered given the retrieved context?
    human_reasoning: str  # Why is this answerable/not answerable?


@dataclass
class ThresholdCalibrationResult:
    """Result of threshold calibration for one operation."""
    operation: OperationType
    optimal_threshold: float
    balanced_accuracy: float
    precision_at_threshold: float
    recall_at_threshold: float
    auc_roc: float
    auc_pr: float
    labeled_samples: int


class ESSThresholdCalibrator:
    """Calibrate ESS thresholds per operation using labeled validation data."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        self.sanity_pyramid = SanityPyramid()
        self.labeled_queries: List[LabeledQuery] = []
        self.calibration_results: Dict[str, ThresholdCalibrationResult] = {}
    
    async def create_labeled_validation_set(self, size_per_operation: int = 20) -> List[LabeledQuery]:
        """
        Create labeled validation set for threshold calibration.
        
        In production, this would involve human labeling. For demonstration,
        we create diverse examples with simulated ground truth.
        """
        logger.info(f"📝 Creating labeled validation set ({size_per_operation} per operation)")
        
        labeled_queries = []
        
        for operation in OperationType:
            operation_queries = await self._create_operation_examples(operation, size_per_operation)
            labeled_queries.extend(operation_queries)
        
        self.labeled_queries = labeled_queries
        
        # Save labeled set
        labeled_data = []
        for lq in labeled_queries:
            labeled_data.append({
                'query_id': lq.query_id,
                'query': lq.query,
                'operation': lq.operation.value,
                'ess_score': lq.ess_score,
                'is_answerable': lq.is_answerable,
                'human_reasoning': lq.human_reasoning
            })
        
        labeled_file = self.work_dir / "labeled_validation_set.json"
        with open(labeled_file, 'w') as f:
            json.dump(labeled_data, f, indent=2)
        
        logger.info(f"✅ Created {len(labeled_queries)} labeled queries, saved to {labeled_file}")
        return labeled_queries
    
    async def _create_operation_examples(self, operation: OperationType, count: int) -> List[LabeledQuery]:
        """Create diverse examples for one operation type."""
        examples = []
        
        for i in range(count):
            # Create diverse scenarios with different ESS characteristics
            if operation == OperationType.LOCATE:
                examples.extend(await self._create_locate_examples(i))
            elif operation == OperationType.EXTRACT:
                examples.extend(await self._create_extract_examples(i))
            elif operation == OperationType.EXPLAIN:
                examples.extend(await self._create_explain_examples(i))
            elif operation == OperationType.COMPOSE:
                examples.extend(await self._create_compose_examples(i))
            elif operation == OperationType.TRANSFORM:
                examples.extend(await self._create_transform_examples(i))
        
        return examples[:count]  # Return exactly `count` examples
    
    async def _create_locate_examples(self, index: int) -> List[LabeledQuery]:
        """Create locate operation examples with varied ESS scores."""
        return [
            # High ESS, should be answerable
            LabeledQuery(
                query_id=f"locate_high_{index}",
                query="find BaseModel class definition",
                operation=OperationType.LOCATE,
                retrieved_chunks=[
                    {'file_path': 'pydantic/main.py', 'content': 'class BaseModel(metaclass=ModelMetaclass):\n    """Base model class"""', 'score': 0.95},
                    {'file_path': 'pydantic/types.py', 'content': 'from .main import BaseModel', 'score': 0.3}
                ],
                gold_data={'gold_paths': ['pydantic/main.py'], 'gold_spans': [('pydantic/main.py', 1, 30)]},
                ess_score=0.85,  # High ESS
                is_answerable=True,
                human_reasoning="Gold path present, exact class definition in context, strong key token match"
            ),
            # Low ESS, not answerable
            LabeledQuery(
                query_id=f"locate_low_{index}",
                query="find ValidationError class definition", 
                operation=OperationType.LOCATE,
                retrieved_chunks=[
                    {'file_path': 'pydantic/utils.py', 'content': 'Some utility functions here', 'score': 0.2},
                    {'file_path': 'tests/test_main.py', 'content': 'def test_validation(): pass', 'score': 0.1}
                ],
                gold_data={'gold_paths': ['pydantic/errors.py'], 'gold_spans': [('pydantic/errors.py', 50, 80)]},
                ess_score=0.15,  # Very low ESS
                is_answerable=False,
                human_reasoning="Gold path not in retrieval, no relevant content, key tokens missing"
            )
        ]
    
    async def _create_extract_examples(self, index: int) -> List[LabeledQuery]:
        """Create extract operation examples."""
        return [
            # High ESS, extractable
            LabeledQuery(
                query_id=f"extract_high_{index}",
                query="show me the validation method signature",
                operation=OperationType.EXTRACT,
                retrieved_chunks=[
                    {'file_path': 'pydantic/main.py', 'content': 'def validate(self, value: Any, field: Field) -> Any:\n    """Validate field value"""\n    return validated_value', 'score': 0.9}
                ],
                gold_data={'gold_paths': ['pydantic/main.py'], 'gold_spans': [('pydantic/main.py', 10, 50)]},
                ess_score=0.82,
                is_answerable=True,
                human_reasoning="Exact method signature present in context, spans match perfectly"
            ),
            # Medium ESS, borderline
            LabeledQuery(
                query_id=f"extract_medium_{index}",
                query="show me error handling code",
                operation=OperationType.EXTRACT,
                retrieved_chunks=[
                    {'file_path': 'pydantic/main.py', 'content': 'try:\n    validate_data()\nexcept Exception as e:\n    # Some handling', 'score': 0.6}
                ],
                gold_data={'gold_paths': ['pydantic/errors.py'], 'gold_spans': [('pydantic/errors.py', 20, 60)]},
                ess_score=0.55,
                is_answerable=False,
                human_reasoning="Related content but not the specific error handling from gold file"
            )
        ]
    
    async def _create_explain_examples(self, index: int) -> List[LabeledQuery]:
        """Create explain operation examples."""
        return [
            # High ESS, explainable
            LabeledQuery(
                query_id=f"explain_high_{index}",
                query="how does pydantic validation work",
                operation=OperationType.EXPLAIN,
                retrieved_chunks=[
                    {'file_path': 'pydantic/main.py', 'content': 'Validation happens in stages: 1) Type checking 2) Field validation 3) Model validation', 'score': 0.8},
                    {'file_path': 'pydantic/validators.py', 'content': 'Each validator runs in sequence, transforming values', 'score': 0.7}
                ],
                gold_data={'gold_paths': ['pydantic/main.py', 'pydantic/validators.py']},
                ess_score=0.72,
                is_answerable=True,
                human_reasoning="Multiple relevant contexts that together explain the validation process"
            ),
            # Low ESS, not explainable
            LabeledQuery(
                query_id=f"explain_low_{index}",
                query="how does async validation work", 
                operation=OperationType.EXPLAIN,
                retrieved_chunks=[
                    {'file_path': 'pydantic/fields.py', 'content': 'Field configuration options', 'score': 0.3}
                ],
                gold_data={'gold_paths': ['pydantic/async_validators.py']},
                ess_score=0.25,
                is_answerable=False,
                human_reasoning="No async validation context retrieved, insufficient information"
            )
        ]
    
    async def _create_compose_examples(self, index: int) -> List[LabeledQuery]:
        """Create compose operation examples."""
        return [
            LabeledQuery(
                query_id=f"compose_high_{index}",
                query="how do BaseModel and Field work together",
                operation=OperationType.COMPOSE,
                retrieved_chunks=[
                    {'file_path': 'pydantic/main.py', 'content': 'class BaseModel: uses Field definitions for validation', 'score': 0.8},
                    {'file_path': 'pydantic/fields.py', 'content': 'Field() specifies validation rules used by BaseModel', 'score': 0.7}
                ],
                gold_data={'gold_paths': ['pydantic/main.py', 'pydantic/fields.py']},
                ess_score=0.75,
                is_answerable=True,
                human_reasoning="Both components present with relationship context"
            )
        ]
    
    async def _create_transform_examples(self, index: int) -> List[LabeledQuery]:
        """Create transform operation examples."""
        return [
            LabeledQuery(
                query_id=f"transform_high_{index}",
                query="convert this model definition to usage example",
                operation=OperationType.TRANSFORM,
                retrieved_chunks=[
                    {'file_path': 'pydantic/main.py', 'content': 'class User(BaseModel):\n    name: str\n    age: int', 'score': 0.7}
                ],
                gold_data={'gold_paths': ['examples/usage.py']},
                ess_score=0.68,
                is_answerable=True,
                human_reasoning="Model definition present, can generate usage example"
            )
        ]
    
    async def calibrate_thresholds(self, labeled_queries: List[LabeledQuery] = None) -> Dict[str, ThresholdCalibrationResult]:
        """Calibrate optimal ESS thresholds per operation using ROC/PR analysis."""
        logger.info("🎯 Calibrating ESS thresholds per operation")
        
        if labeled_queries is None:
            labeled_queries = self.labeled_queries
        
        if not labeled_queries:
            raise ValueError("No labeled queries available. Run create_labeled_validation_set first.")
        
        # Group by operation
        by_operation = {}
        for lq in labeled_queries:
            op = lq.operation.value
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(lq)
        
        calibration_results = {}
        
        for op_name, op_queries in by_operation.items():
            logger.info(f"📊 Calibrating {op_name} threshold ({len(op_queries)} samples)")
            
            # Extract ESS scores and labels
            ess_scores = np.array([lq.ess_score for lq in op_queries])
            labels = np.array([lq.is_answerable for lq in op_queries])
            
            if len(np.unique(labels)) < 2:
                logger.warning(f"⚠️ Skipping {op_name} - need both positive and negative examples")
                continue
            
            # Calculate ROC curve
            fpr, tpr, roc_thresholds = roc_curve(labels, ess_scores)
            roc_auc = np.trapz(tpr, fpr)
            
            # Calculate PR curve
            precision, recall, pr_thresholds = precision_recall_curve(labels, ess_scores)
            pr_auc = np.trapz(precision, recall)
            
            # Find optimal threshold using balanced accuracy
            best_threshold = 0.5
            best_balanced_accuracy = 0.0
            best_precision = 0.0
            best_recall = 0.0
            
            for threshold in np.linspace(0.1, 0.9, 81):  # Test thresholds from 0.1 to 0.9
                predictions = ess_scores >= threshold
                balanced_acc = balanced_accuracy_score(labels, predictions)
                
                if balanced_acc > best_balanced_accuracy:
                    best_balanced_accuracy = balanced_acc
                    best_threshold = threshold
                    
                    # Calculate precision and recall at this threshold
                    tp = np.sum((predictions == 1) & (labels == 1))
                    fp = np.sum((predictions == 1) & (labels == 0))
                    fn = np.sum((predictions == 0) & (labels == 1))
                    
                    best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Create calibration result
            result = ThresholdCalibrationResult(
                operation=OperationType(op_name),
                optimal_threshold=best_threshold,
                balanced_accuracy=best_balanced_accuracy,
                precision_at_threshold=best_precision,
                recall_at_threshold=best_recall,
                auc_roc=roc_auc,
                auc_pr=pr_auc,
                labeled_samples=len(op_queries)
            )
            
            calibration_results[op_name] = result
            
            logger.info(f"✅ {op_name}: threshold={best_threshold:.2f}, balanced_acc={best_balanced_accuracy:.2f}, P={best_precision:.2f}, R={best_recall:.2f}")
            
            # Generate ROC/PR plots
            await self._plot_calibration_curves(op_name, ess_scores, labels, roc_thresholds, pr_thresholds, fpr, tpr, precision, recall, best_threshold)
        
        self.calibration_results = calibration_results
        
        # Save calibration results
        await self._save_calibration_results(calibration_results)
        
        return calibration_results
    
    async def _plot_calibration_curves(self, operation: str, ess_scores: np.ndarray, labels: np.ndarray,
                                      roc_thresholds: np.ndarray, pr_thresholds: np.ndarray,
                                      fpr: np.ndarray, tpr: np.ndarray, 
                                      precision: np.ndarray, recall: np.ndarray,
                                      optimal_threshold: float):
        """Generate ROC and PR curves for threshold calibration."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {np.trapz(tpr, fpr):.2f})')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{operation.title()} - ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal threshold
        predictions = ess_scores >= optimal_threshold
        fpr_opt = np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)
        tpr_opt = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
        ax1.plot(fpr_opt, tpr_opt, 'ro', markersize=8, label=f'Optimal τ={optimal_threshold:.2f}')
        
        # PR Curve
        ax2.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {np.trapz(precision, recall):.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{operation.title()} - Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal threshold
        predictions = ess_scores >= optimal_threshold
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ax2.plot(recall_opt, precision_opt, 'ro', markersize=8, label=f'Optimal τ={optimal_threshold:.2f}')
        
        plt.tight_layout()
        
        plot_path = self.work_dir / f"calibration_curves_{operation}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Calibration curves saved: {plot_path}")
    
    async def _save_calibration_results(self, results: Dict[str, ThresholdCalibrationResult]):
        """Save calibration results to disk."""
        results_data = {}
        for op_name, result in results.items():
            results_data[op_name] = {
                'optimal_threshold': result.optimal_threshold,
                'balanced_accuracy': result.balanced_accuracy,
                'precision_at_threshold': result.precision_at_threshold,
                'recall_at_threshold': result.recall_at_threshold,
                'auc_roc': result.auc_roc,
                'auc_pr': result.auc_pr,
                'labeled_samples': result.labeled_samples
            }
        
        results_file = self.work_dir / "ess_threshold_calibration_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Generate calibrated thresholds config
        thresholds_config = {
            'ess_thresholds': {op_name: result.optimal_threshold for op_name, result in results.items()},
            'calibration_metadata': {
                'calibration_date': str(Path().resolve()),
                'total_labeled_samples': sum(r.labeled_samples for r in results.values()),
                'calibration_quality': {
                    op_name: {
                        'balanced_accuracy': result.balanced_accuracy,
                        'samples': result.labeled_samples
                    }
                    for op_name, result in results.items()
                }
            }
        }
        
        config_file = self.work_dir / "calibrated_ess_thresholds.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(thresholds_config, f, indent=2)
        
        logger.info(f"💾 Calibration results saved:")
        logger.info(f"   JSON: {results_file}")
        logger.info(f"   YAML: {config_file}")
    
    def apply_calibrated_thresholds(self) -> ESS_Thresholds:
        """Apply calibrated thresholds to create new ESS_Thresholds object."""
        if not self.calibration_results:
            logger.warning("No calibration results available. Using default thresholds.")
            return ESS_Thresholds()
        
        # Extract thresholds
        thresholds = {}
        for op_name, result in self.calibration_results.items():
            thresholds[op_name] = result.optimal_threshold
        
        calibrated_thresholds = ESS_Thresholds(
            locate=thresholds.get('locate', 0.8),
            extract=thresholds.get('extract', 0.75),
            explain=thresholds.get('explain', 0.6),
            compose=thresholds.get('compose', 0.7),
            transform=thresholds.get('transform', 0.65)
        )
        
        logger.info("🎯 Applied calibrated ESS thresholds:")
        for op_type in OperationType:
            threshold = getattr(calibrated_thresholds, op_type.value)
            logger.info(f"   {op_type.value}: {threshold:.2f}")
        
        return calibrated_thresholds


async def main():
    """Run ESS threshold calibration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    calibrator = ESSThresholdCalibrator(Path('ess_calibration_results'))
    
    # Create labeled validation set
    labeled_queries = await calibrator.create_labeled_validation_set(size_per_operation=8)
    print(f"📋 Created {len(labeled_queries)} labeled queries")
    
    # Calibrate thresholds
    results = await calibrator.calibrate_thresholds()
    
    # Apply calibrated thresholds
    calibrated = calibrator.apply_calibrated_thresholds()
    
    print(f"\n🎯 CALIBRATION COMPLETE")
    print(f"Operations calibrated: {len(results)}")
    print(f"Total labeled samples: {sum(r.labeled_samples for r in results.values())}")
    
    print(f"\n📊 CALIBRATED THRESHOLDS:")
    for op_type in OperationType:
        threshold = getattr(calibrated, op_type.value)
        if op_type.value in results:
            result = results[op_type.value]
            print(f"  {op_type.value:9}: {threshold:.2f} (bal_acc: {result.balanced_accuracy:.2f}, samples: {result.labeled_samples})")
        else:
            print(f"  {op_type.value:9}: {threshold:.2f} (default)")


if __name__ == "__main__":
    asyncio.run(main())