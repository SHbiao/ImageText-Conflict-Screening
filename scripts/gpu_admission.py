"""Read-only admission: busy GPU jobs are deferred, never preempted."""
import subprocess


def check_idle(min_free_mib=6000):
    commands = [['nvidia-smi'], ['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv'],
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv']]
    outputs = [subprocess.check_output(c, text=True) for c in commands]
    print('\n'.join(outputs))
    if len(outputs[2].strip().splitlines()) > 1:
        raise SystemExit('WAITING_FOR_GPU: active compute process; resume after resources are idle')
    # Confirm process absence twice; utilization alone never grants admission.
    import time
    time.sleep(2)
    repeat = subprocess.check_output(commands[2], text=True)
    if len(repeat.strip().splitlines()) > 1:
        raise SystemExit('WAITING_FOR_GPU: compute process appeared during admission check')
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    if int(memory.splitlines()[0]) < min_free_mib:
        raise SystemExit(f'WAITING_FOR_GPU: less than {min_free_mib} MiB free; retain the queued command')
