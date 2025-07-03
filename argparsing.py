import argparse
import deformations as defs

def_names = ["HomogDef", "TorusDef", "ProjDef", "SkewMatDef", "FlowDef"]


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="CP(N) dimension", type=int, default=3)
    parser.add_argument("--i", help="OnePointFn element i",
                        type=int, default=0)
    parser.add_argument("--j", help="OnePointFn element j",
                        type=int, default=0)
    parser.add_argument(
        "--pidx", help="particle idx (0 | 1)", type=int, default=0)
    parser.add_argument(
        "--beta", help="ToyModelAction parameter beta", type=float, default=4.5)
    parser.add_argument("--epochs", help="epochs", type=int, default=1000)
    parser.add_argument("--batch_size", help="batch size",
                        type=int, default=1000)
    parser.add_argument("--split", help="training to test data split ratio",
                        type=float, default=0.8)
    parser.add_argument("--deftype", help=f"constant deformation type ({'|'.join(def_names)})",
                        type=str, default="HomogDef")
    parser.add_argument("--hidden_dim", help=f"parnet hidden layer dimension",
                        type=int, default=16)
    parser.add_argument(
        "--update", help="print updates every 10 epochs", type=bool, default=False)
    args = parser.parse_args()

    assert args.deftype in def_names, f"Deformation type not recognized, choose from {', '.join(def_names)}"

    if args.deftype == "TorusDef":
        deformation = defs.TorusDeformations
    elif args.deftype == "ProjDef":
        deformation = defs.ProjectorDeformations
    elif args.deftype == "SkewMatDef":
        deformation = defs.SkewMatrixDeformations
    elif args.deftype == "FlowDef":
        deformation = defs.FlowDeformations
    else:
        deformation = defs.HomogenousDeformations

    return args, deformation
