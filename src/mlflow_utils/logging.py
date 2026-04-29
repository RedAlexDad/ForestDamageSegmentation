"""MLflow utilities for logging and tracking experiments."""

import os
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import pandas as pd


class MLflowLogger:
    """Класс для логирования экспериментов в MLflow."""
    
    def __init__(
        self,
        experiment_name: str = "forest-damage-segmentation",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Инициализация MLflow логгера.
        
        Args:
            experiment_name: Название эксперимента
            tracking_uri: URI для MLflow сервера (None = локальный)
            run_name: Название запуска (опционально)
            tags: Теги для запуска (опционально)
        """
        self.experiment_name = experiment_name
        
        # Настройка tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Установка эксперимента
        mlflow.set_experiment(experiment_name)
        
        # Запуск эксперимента
        self.active_run = mlflow.start_run(
            run_name=run_name,
            tags=tags
        )
        self.run_id = self.active_run.info.run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Логирование параметров."""
        mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Логирование одной метрики."""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Логирование нескольких метрик."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_figure(self, fig: plt.Figure, name: str) -> None:
        """Логирование matplotlib figure."""
        mlflow.log_figure(fig, name)
    
    def log_image(self, image: np.ndarray, name: str, format: str = "png") -> None:
        """Логирование изображения."""
        # Сохранение временного файла
        import tempfile
        import imageio
        
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
            imageio.imwrite(tmp.name, image)
            mlflow.log_artifact(tmp.name, name)
            os.unlink(tmp.name)
    
    def log_images_grid(
        self,
        images: List[np.ndarray],
        titles: Optional[List[str]] = None,
        name: str = "images_grid.png",
        cols: int = 4
    ) -> None:
        """Логирование сетки изображений."""
        n = len(images)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for i, img in enumerate(images):
            row, col = i // cols, i % cols
            axes[row, col].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[row, col].axis('off')
            if titles and i < len(titles):
                axes[row, col].set_title(titles[i])
        
        # Удаление пустых subplots
        for i in range(n, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        self.log_figure(fig, name)
        plt.close(fig)
    
    def log_model(
        self,
        model: tf.keras.Model,
        name: str = "model",
        saved_model_format: bool = True
    ) -> None:
        """Логирование Keras/TF модели."""
        if saved_model_format:
            mlflow.keras.log_model(model, name)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                model.save(tmp.name)
                mlflow.log_artifact(tmp.name, f"{name}.h5")
                os.unlink(tmp.name)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Логирование артефакта."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict_as_json(self, data: Dict, name: str) -> None:
        """Логирование словаря как JSON."""
        import json
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as tmp:
            json.dump(data, tmp, indent=2)
            mlflow.log_artifact(tmp.name, name)
            os.unlink(tmp.name)
    
    def log_training_history(self, history) -> None:
        """Логирование истории обучения."""
        # Метрики
        metrics = {}
        for key, values in history.history.items():
            final_value = values[-1]
            metrics[key] = final_value
            # Логируем каждую эпоху
            for epoch, value in enumerate(values):
                mlflow.log_metric(key, value, step=epoch)
        
        # Графики
        fig = self.plot_training_curves(history)
        self.log_figure(fig, "training_curves.png")
        plt.close(fig)
        
        return metrics
    
    def plot_training_curves(
        self,
        history,
        metrics: List[str] = None
    ) -> plt.Figure:
        """Построение графиков обучения."""
        if metrics is None:
            metrics = ['loss', 'dice_coefficient', 'iou']
        
        h = history.history
        n_metrics = len([m for m in metrics if m in h])
        
        if n_metrics == 0:
            n_metrics = 1
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        idx = 0
        for metric in metrics:
            if metric in h:
                axes[idx].plot(h[metric], label=f'Train {metric}')
                if f'val_{metric}' in h:
                    axes[idx].plot(h[f'val_{metric}'], label=f'Val {metric}')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric)
                axes[idx].set_title(f'{metric} over epochs')
                axes[idx].legend()
                axes[idx].grid(True)
                idx += 1
        
        plt.tight_layout()
        return fig
    
    def log_predictions(
        self,
        images: np.ndarray,
        masks_true: np.ndarray,
        masks_pred: np.ndarray,
        names: Optional[List[str]] = None,
        max_samples: int = 8
    ) -> None:
        """Логирование предиктов (изображение + маска + предсказание)."""
        n = min(max_samples, len(images))
        
        for i in range(n):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Входное изображение
            if images[i].ndim == 3 and images[i].shape[-1] in [1, 3]:
                axes[0].imshow(images[i][..., :3] if images[i].shape[-1] == 3 else images[i][..., 0])
            else:
                axes[0].imshow(images[i])
            axes[0].set_title('Input')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(masks_true[i], cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(masks_pred[i], cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            name = f"prediction_{names[i] if names and i < len(names) else i}.png"
            self.log_figure(fig, f"predictions/{name}")
            plt.close(fig)
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str = "confusion_matrix.png"
    ) -> Dict[str, float]:
        """Логирование confusion matrix."""
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        
        # Расчёт метрик
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "true_positives": float(tp),
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # График
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        self.log_figure(fig, name)
        plt.close(fig)
        
        return metrics
    
    def log_dataset_statistics(
        self,
        dataset,
        name: str = "dataset_stats"
    ) -> None:
        """Логирование статистики датасета."""
        stats = {
            "n_samples": len(dataset) if hasattr(dataset, '__len__') else 0,
            "n_classes": getattr(dataset, 'n_classes', 1),
            "image_shape": getattr(dataset, 'image_shape', None),
        }
        
        self.log_dict_as_json(stats, f"{name}.json")
        
        # Визуализация распределения классов
        if hasattr(dataset, 'get_class_distribution'):
            class_dist = dataset.get_class_distribution()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(class_dist.keys(), class_dist.values())
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
            plt.tight_layout()
            self.log_figure(fig, f"{name}_class_dist.png")
            plt.close(fig)
    
    def end_run(self, status: str = "FINISHED") -> None:
        """Завершение запуска."""
        mlflow.end_run(status=status)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()


class ExperimentTracker:
    """К��ас�� для отслеживания и сравнения экспериментов."""
    
    def __init__(self, experiment_name: str = "forest-damage-segmentation"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
    
    def get_experiment(self):
        """Получить эксперимент."""
        return mlflow.get_experiment_by_name(self.experiment_name)
    
    def get_runs(self, max_results: int = 100):
        """Получить все запуски эксперимента."""
        exp = self.get_experiment()
        if exp is None:
            mlflow.create_experiment(self.experiment_name)
            exp = self.get_experiment()
        
        return self.client.search_runs(
            [exp.experiment_id],
            "attributes.status != 'DELETED'",
            max_results=max_results
        )
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """Сравнение метрик разных запусков."""
        import pandas as pd
        
        data = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            row = {"run_id": run_id}
            for metric in metrics:
                if f"metrics.{metric}" in run.data.metrics:
                    row[metric] = run.data.metrics[f"metrics.{metric}"]
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_run(self, metric: str, maximize: bool = True) -> Optional[str]:
        """Получить лучший run по метрике."""
        runs = self.get_runs()
        
        best_run = None
        best_value = float('-inf') if maximize else float('inf')
        
        for run in runs:
            if f"metrics.{metric}" in run.data.metrics:
                value = run.data.metrics[f"metrics.{metric}"]
                if (maximize and value > best_value) or (not maximize and value < best_value):
                    best_value = value
                    best_run = run.info.run_id
        
        return best_run
    
    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str):
        """Скачать артефакты из run."""
        self.client.download_artifacts(run_id, artifact_path, dst_path)
    
    def register_model(
        self,
        run_id: str,
        model_path: str,
        model_name: str
    ) -> None:
        """Регистрация модели в Model Registry."""
        result = mlflow.register_model(
            f"runs:/{run_id}/{model_path}",
            model_name
        )
        return result


def setup_mlflow(
    experiment_name: str = "forest-damage-segmentation",
    tracking_uri: Optional[str] = None
) -> MLflowLogger:
    """Упрощённая настройка MLflow."""
    return MLflowLogger(experiment_name, tracking_uri)


# Callback для Keras с MLflow логированием
class MLOgflowCallback(tf.keras.callbacks.Callback):
    """Callback для автоматического логирования в MLflow."""
    
    def __init__(self, logger: MLflowLogger, log_frequency: int = 1):
        super().__init__()
        self.logger = logger
        self.log_frequency = log_frequency
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and (epoch + 1) % self.log_frequency == 0:
            metrics = {f"train_{k}": v for k, v in logs.items()}
            self.logger.log_metrics(metrics, step=epoch)
    
    def on_train_end(self, logs=None):
        if logs:
            final_metrics = {f"final_{k}": v for k, v in logs.items()}
            self.logger.log_metrics(final_metrics)