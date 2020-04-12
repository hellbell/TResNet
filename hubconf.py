dependencies = ['torch']

from src.models.tresnet import TResnetM, TResnetL, TResnetXL

tresnet_model_urls = {
  'tresnet_m': 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m.pth',
  'tresnet_m_448': 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m_448.pth',
}

# | [tresnet_m.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m.pth) | 224 |
# | [tresnet_m_448.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m_448.pth) | 448 |
# | [tresnet_l.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_l.pth) | 224 |
# | [tresnet_l_448.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_l_448.pth) | 448 |
# | [tresnet_xl.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_xl.pth) | 224 |
# | [tresnet_xl_448.pth](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_xl_448.pth) | 448 |

def tresnet_m(pretrained=False, num_classes=1000, remove_aa_jit=False):
  model = TResnetM({
    'num_classes': num_classes,
    'remove_aa_jit': remove_aa_jit
  })
  
  if pretrained:
    model.load_state_dict(torch.hub.load_state_dict_from_url(
      tresnet_model_urls['tresnet_m'], progress=True, check_hash=True))
  
  return model 
  
    
    
