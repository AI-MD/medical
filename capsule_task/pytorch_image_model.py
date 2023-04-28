import timm

m = timm.create_model('mobilenetv3_large_100', pretrained=True)
m.eval()