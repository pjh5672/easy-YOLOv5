from loss.v5loss import YOLOv5Loss


def build_criterion(**kwargs):
    model = kwargs.get('model')
    device = kwargs.get('device')
    imgsz = kwargs.get('imgsz')
    return YOLOv5Loss(model=model, device=device, imgsz=imgsz)