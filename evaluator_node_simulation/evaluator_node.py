import json
import random
import time
import logging
from queue import PriorityQueue

# Configure logging
logging.basicConfig(
    filename="node.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Base interface for evaluation metrics
class EvaluationMetric:
    def evaluate(self, inputs, ground_truth):
        raise NotImplementedError("This method should be implemented by subclasses")

# Implementation of MeanSquaredError metric
class MeanSquaredError(EvaluationMetric):
    def evaluate(self, inputs, ground_truth):
        if len(inputs) != len(ground_truth):
            raise ValueError("Inputs and ground truth must have the same length")
        n = len(inputs)
        return sum((i - g) ** 2 for i, g in zip(inputs, ground_truth)) / n

# Task structure for the PriorityQueue
class Task:
    def __init__(self, task_id, inputs, ground_truth, metric, priority=1):
        self.task_id = task_id
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.metric = metric
        self.priority = priority  # Lower values indicate higher priority

    def __lt__(self, other):
        return self.priority < other.priority

# Evaluator Node
class EvaluatorNode:
    def __init__(self, queue_size_limit=10):
        self.task_queue = PriorityQueue()
        self.queue_size_limit = queue_size_limit
        self.metric_mapping = {"MeanSquaredError": MeanSquaredError()}

    def simulate_resources(self):
        """Simulate CPU and memory usage."""
        cpu_usage = random.randint(50, 100)
        memory_usage = random.randint(50, 100)
        return cpu_usage, memory_usage

    def add_task(self, task):
        """Add a task to the queue, respecting the queue size limit."""
        if self.task_queue.qsize() >= self.queue_size_limit:
            logging.warning(
                f"Task {task.task_id} rejected due to queue size limit."
            )
            return False
        self.task_queue.put(task)
        return True

    def process_task(self, task):
        """Process a task using the specified metric."""
        try:
            cpu_usage, memory_usage = self.simulate_resources()
            if cpu_usage > 80 or memory_usage > 80:
                logging.warning(
                    f"Task {task.task_id} rejected due to resource constraints: CPU={cpu_usage}%, Memory={memory_usage}%."
                )
                return False

            metric = self.metric_mapping.get(task.metric)
            if not metric:
                logging.error(f"Task {task.task_id} failed. Unknown metric: {task.metric}")
                return False

            result = metric.evaluate(task.inputs, task.ground_truth)
            logging.info(f"Task {task.task_id} processed successfully. Result: {result}")
            return True

        except Exception as e:
            logging.error(f"Task {task.task_id} failed. Reason: {str(e)}")
            return False

    def retry_task(self, task, retries=3, backoff=2):
        """Retry a task with exponential backoff."""
        for attempt in range(retries):
            success = self.process_task(task)
            if success:
                return
            time.sleep(backoff ** attempt)  # Exponential backoff
        logging.error(f"Task {task.task_id} permanently failed after {retries} retries.")

    def run(self, tasks_file):
        """Run the evaluator node by reading tasks from a file."""
        with open(tasks_file, "r") as f:
            tasks = json.load(f)

        for task_data in tasks:
            task = Task(
                task_id=task_data["task_id"],
                inputs=task_data["inputs"],
                ground_truth=task_data["ground_truth"],
                metric=task_data["metric"],
                priority=task_data.get("priority", 1),
            )

            if not self.add_task(task):
                continue

        while not self.task_queue.empty():
            task = self.task_queue.get()
            self.retry_task(task)

# Main script
if __name__ == "__main__":
    evaluator = EvaluatorNode(queue_size_limit=10)
    evaluator.run("tasks.json")
