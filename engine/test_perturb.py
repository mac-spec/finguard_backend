from engine.simulator_and_model import generate_normal_fraud_txn
from engine.perturb import generate_perturbations, MANIPULABLE
from engine.simulator_and_model import IMMUTABLE

def main():
    sample = generate_normal_fraud_txn()
    runs = generate_perturbations(sample, n=5, seed=42)
    print("run | changed fields")
    for i, row in enumerate(runs, 1):
        changes = [f"{k}: {sample[k]} -> {row[k]}" for k in MANIPULABLE if sample[k] != row[k]]
        print(f"{i:>3} | " + ("; ".join(changes) or "none"))
        for key in IMMUTABLE:
            assert row[key] == sample[key], f"Immutable feature changed: {key}"
        for key in sample.index:
            if key not in MANIPULABLE and row[key] != sample[key]:
                raise AssertionError(f"Non-manipulable field changed: {key}")
    print("PASS: immutable and all non-manipulable fields remained identical.")

if __name__ == "__main__":
    main()
