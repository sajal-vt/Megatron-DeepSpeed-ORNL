import logging
import os
import subprocess
import sys
import dh_utils
import parse
from deephyper.evaluator import profile

# Retrieve constants
NNODES = int(os.environ["NNODES"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NGPUS_PER_TRAINING = int(os.environ["NGPUS_PER_TRAINING"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
DEEPHYPER_DB_HOST = os.environ["DEEPHYPER_DB_HOST"]


def _parse_results(job_id):
    # TODO: adapt to your needs (requires `pip install parse` package)
    with open(str(DEEPHYPER_LOG_DIR)+f'/output_{job_id}.txt','rb') as f:
        text = f.read()
    print("text=", text)
    #TFLOPs: 6.93
    #samples per second: 59.501
    res = parse.search("TFLOPs: {:f}", text.decode())
    samples = parse.search("samples per second: {:f}", text.decode())
    if res:
        return res[0],samples[0]
    else:
        return "F","F"


# Definition of the Black-Box
@profile  # will collect "start/end"-times of the function
def run_distributed_training(job, dequed=None):
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "train.py")

    # TODO: Launch a subprocess with `srun` to train neural networks
    params = job.parameters
    #time srun -u -n32 -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest
    prefix = "".join([f"srun", f" -N {NGPUS_PER_TRAINING//8} -n {NGPUS_PER_TRAINING}" ,
            f" --ntasks-per-node=8 --gpus-per-node=8",
            f" --cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind threads",
            f" --gpus-per-task=1 --gpu-bind=closest"])
    command,job_id = dh_utils.create_launch_command(prefix, params, job.id, dequed, DEEPHYPER_LOG_DIR) 
    print("Command = ", command)
    
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        #subprocess.run(command, shell=True, stderr=subprocess.STDOUT)
        print("result =", result, flush=True)
        output = "F"
        samples = "F"
        try:
            #output = _parse_results(result)
            output, samples = _parse_results(job_id)
        except Exception as excp:
            print(excp)
            output = "F"
    except Exception as excp:
        print(excp) 
        ouput = "F"
        samples = "F"

    print("Got the output", output)
    #output = 35.0
    #! Maximization of objective
    objective = output
    print(objective)
    # Some other infos can be collected (needs to be JSON serializable)
    metadata = {"samples": samples}

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from dh_utils import read_node_list

    
    # Setup info-level logging (informative can be deactivated later)
    logging.basicConfig(
        filename=os.path.join(DEEPHYPER_LOG_DIR, f"deephyper.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - "
        + "%(message)s",
        force=True,
    )

    # Define the problem
    problem = HpProblem()
    problem.add_hyperparameter((0, 3), "tp", default_value=0) 
    problem.add_hyperparameter((0, 3), "pp", default_value=0)
    problem.add_hyperparameter((10, 20), "mbs", default_value=10)
    problem.add_hyperparameter((10, 160), "gbs", default_value=10)
    
    # Create the node queue
    queue, _ = read_node_list()
    print("The queue:", queue, len(queue))
    print(NTOTGPUS, NGPUS_PER_TRAINING, NTOTGPUS // NGPUS_PER_TRAINING, len(queue))
    #assert NTOTGPUS // NGPUS_PER_TRAINING == len(queue)
    evaluator = queued(ProcessPoolEvaluator)(
            run_distributed_training,
            num_workers = NTOTGPUS // NGPUS_PER_TRAINING,
            queue = queue,
            queue_pop_per_task=4 #Remove the hard-coded value later
            )
    '''
    # Define The Evaluator (scheduling `run_distributed_training` tasks)
    evaluator = Evaluator.create(
        run_distributed_training,
        method="process",
        method_kwargs={
            "num_workers": NTOTGPUS // NGPUS_PER_TRAINING,
        },
    )
    '''
    # Define the search method and scalarization
    search = CBO(
        problem,
        evaluator,
        acq_func="UCB",
        multi_point_strategy="cl_min", # Constant liar strategy
        random_state=42,
        # Location where to store the results
        log_dir=DEEPHYPER_LOG_DIR,
        # Number of threads used to update surrogate model of BO
        n_jobs=OMP_NUM_THREADS,
    )

    # Run the search with a limit of 10_000 evaluations and a timeout of 30 minutes
    results = search.search(max_evals=1000, timeout=14400)
