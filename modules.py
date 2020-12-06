def printProgressBar(iteration, total, prefix='', suffix='', decimals=2, length=80, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


class AverageMeter(object):
    def __init__(self):
        self.number = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, value):
        self.number += 1.
        self.sum += value
        self.avg = self.sum / self.number

    def reset(self):
        self.number, self.sum, self.avg = 0., 0., 0.


def precision_recall(dt, gt, t=0.5, is_binary=True):
    """
    :param dt: 1D tensor of detections in {0,1} (if is_binary)
    :param gt: 1D tensor of ground truth values in {0,1}
    :param t: if dt isn't binary, threshold for outputting 1
    :param is_binary: indicates whether dt is in {0,1} or [0,1]
    :return: precision, recall
    """
    if not is_binary:
        dt[dt < t] = 0
        dt[dt >= t] = 1
    tp = sum(dt * gt)
    # tn = sum((1 - dt) * (1 - gt))
    fp = sum((1 - gt) * dt)
    fn = sum(gt * (1 - dt))

    return (tp / (tp + fp)).item(), (tp / (tp + fn)).item()
