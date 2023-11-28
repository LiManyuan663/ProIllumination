from .intrinsic_model import Intrinsics_Model


def create_model(opt, print_info=False):
    model = Intrinsics_Model(opt,print_info)
    print("model [%s] was created" % (model.name()))

    # model.initialize()
    return model
