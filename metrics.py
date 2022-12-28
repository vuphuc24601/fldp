class Metric:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.loss = []

    def update(self, pt, gt, loss):
        """
        pt: n x 1, integer
        gt: n x 1, integer
        loss: float scale
        """

        self.total += len(gt)
        self.correct += (pt == gt).sum().item()
        self.loss.append(loss)

    def get(self):
        """
        return:
            averaged loss
            total correctness
            precision
            recall
            f1
            total samples
        """
        avg_loss = sum(self.loss) / len(self.loss)
        return {
            "loss": avg_loss,
            "correct": self.correct,
            "total": self.total,
        }
