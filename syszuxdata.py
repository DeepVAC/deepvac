from lib.syszux_executor import AugChain
import conf as deepvac_config

if __name__ == "__main__":
    flow = "Retina => ISFace"
    chain = AugChain(flow, deepvac_config)
    print(chain.transforms)