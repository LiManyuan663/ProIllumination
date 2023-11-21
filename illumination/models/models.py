from .intrinsic_model import Intrinsics_Model


def create_model(opt):
    model = Intrinsics_Model(opt)
    print("model [%s] was created" % (model.name()))

    # model.initialize()
    return model
