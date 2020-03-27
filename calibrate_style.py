
import numpy as np
from numpy import pi,sin,cos,arctan
import subprocess

#########################################

args={}

### Commonly changed for art
args['output_image']="out/out.jpg" #[out.png]
args['style_image']="styles/elephant.jpg" #Style target image [examples/inputs/seated-nude.jpg]
args['content_image']="content/carlos.jpg" #Content target image [examples/inputs/tubingen.jpg]
args['style_scale']=1 #[1]
args['style_weight']=100 #[100]
args['content_weight']=5 #[5]
args['init']="random" #random|image [random]
#args['init_image']="" #[]
args['original_colors']=0 #[0]
args['image_size']=512 #Maximum height / width of generated image [512]
args['num_iterations']=601 #[1000]
args['save_iter']=100 #[100]
args['print_iter']=50 #[50]

### I havent touched
args['tv_weight']=0.001 #[0.001]
args['pooling']="max" #max|avg [max]
args['seed']=-1 #[-1]
args['proto_file']="models/VGG_ILSVRC_19_layers_deploy.prototxt" #[models/VGG_ILSVRC_19_layers_deploy.prototxt]
args['model_file']="models/VGG_ILSVRC_19_layers.caffemodel" #[models/VGG_ILSVRC_19_layers.caffemodel]
args['content_layers']="relu4_2" #layers for content [relu4_2]
args['style_layers']="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1" #layers for style [relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
args['lbfgs_num_correction']=0 #[0]
#args['style_blend_weights']="" #[nil]

### Seem to be well tuned
args['optimizer']="lbfgs" #lbfgs|adam [lbfgs]
#args['normalize_gradients']="" #[]
args['learning_rate']=10 #[10]
args['gpu']=0 #Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1 [0]
args['cudnn_autotune']="" #[false]
args['backend']="cudnn" #nn|cudnn|clnn [nn]

def build_args(args):
    return list(filter(None,["th", "neural_style.lua"]+ sum([ "-{} {}".format(key,args[key]).split(" ") for key in args.keys() ], []) ))
def get_name(path):
    return path.split("/")[-1].split(".")[0]
#########################################



max_sw=args['style_weight']
max_cw=args['content_weight']

style=args['style_image'].split("/")[-1].split(".")[0]
content=args['content_image'].split("/")[-1].split(".")[0]

rowlist=[]

for style in filter(None, subprocess.run("ls styles",shell=True,capture_output=True).stdout.decode('utf-8').split("\n")  ):
    style="bubblewrap.jpg"
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    args['style_image']="styles/"+style #Style target image [examples/inputs/seated-nude.jpg]
    suff=style.rstrip(".jpg")+"-"+content+"-calibration.jpg"

    for tan_theta in np.linspace(0.1,0.9,6):
    #    break
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        print(tan_theta)
        theta=arctan(tan_theta)
        args['style_weight'],args['content_weight']=max_sw*sin(theta), max_cw*cos(theta)
        #content_ratio="{0:1.2f}".format(cos(theta))
        tan_theta_str="{0:1.2f}".format(tan_theta)

        # build output filename
        args['output_image']="output/"+tan_theta_str+"-"+suff
        print(build_args(args))
        rowname="{}-row.jpg".format(args['output_image'].rstrip(".jpg") )
        subprocess.run(build_args(args) )
        subprocess.run(\
            ["convert", "+append", "{0}_*.jpg".format( args['output_image'].rstrip(".jpg")   ), rowname]\
            )
        rowlist.append(rowname)
    subprocess.run(\
            ["convert", "-append"]+rowlist+["output/full-{}".format(suff)]\
        )
    subprocess.run("notify-send finished another calibration!",shell=True)
    break

