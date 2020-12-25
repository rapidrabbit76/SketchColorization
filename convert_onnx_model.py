import torch
from models import SketchColorizationModel


def convert_onnx_model(path: str, save_dir: str):
    model_ts = torch.jit.load(path)
    model = SketchColorizationModel()
    model.load_state_dict(model_ts.state_dict())

    sample_data_line = torch.randn(1, 1, 512, 512)
    sample_data_line_draft = torch.randn(1, 1, 128, 128)
    sample_data_hint = torch.randn(1, 4, 128, 128)

    model.eval()

    model.forward(sample_data_line,
                  sample_data_line_draft,
                  sample_data_hint)

    torch.onnx.export(model,
                      (sample_data_line,
                       sample_data_line_draft,
                       sample_data_hint),
                      save_dir,
                      opset_version=11,
                      export_params=True,
                      input_names=['line',
                                   'line_draft',
                                   'hint'],
                      output_names=['colored'])


if __name__ == "__main__":
    model_path = './SketchColorizationModel.zip'
    save_dir = './SketchColorizationModel.onnx'
    convert_onnx_model(model_path, save_dir)
