if __package__:
    from .yolov5 import YOLOv5
else:
    from yolov5 import YOLOv5


MODEL_LIST = ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
              

def build_model(arch_name, num_classes):
    """build method for defined architectures

    Args:
        arch_name (str): classifier name
        num_classes (int): number of classes in prediction

    Returns:
        torch.nn.Module: classifier architecture
    """
    
    arch_name = arch_name.lower()
    assert arch_name in MODEL_LIST, \
        f'not support such architecture, got {arch_name}.'
    return YOLOv5(scale=arch_name[-1], num_classes=num_classes)


if __name__ == '__main__':
    from utils.torch_utils import model_info

    model = build_model(arch_name='yolov5s', num_classes=80)
    model_info(model, input_size=640)