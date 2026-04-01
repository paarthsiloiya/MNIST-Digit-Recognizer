import json
import argparse
import sys

def main(keras_json, tfjs_json, report_json):
    print("Comparing Keras and TFJS inference results!")
    
    try:
        with open(keras_json, 'r') as f:
            keras_data = json.load(f)
        with open(tfjs_json, 'r') as f:
            tfjs_data = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
        
    keras_acc = keras_data['accuracy']
    tfjs_acc = tfjs_data['accuracy']
    keras_preds = keras_data['predicted_labels']
    tfjs_preds = tfjs_data['predicted_labels']
    
    if len(keras_preds) != len(tfjs_preds):
        print("Mismatch in number of samples!")
        sys.exit(1)
        
    n = len(keras_preds)
    matches = sum(1 for i in range(n) if keras_preds[i] == tfjs_preds[i])
    match_fraction = matches / n
    acc_diff = abs(keras_acc - tfjs_acc)
    
    print(f"Keras Accuracy: {keras_acc*100:.2f}%")
    print(f"TFJS Accuracy:  {tfjs_acc*100:.2f}%")
    print(f"Accuracy Diff:  {acc_diff*100:.2f}%")
    print(f"Identical Preds: {match_fraction*100:.2f}%")
    
    # Acceptance Criteria: diff <= 0.005 OR match >= 0.98
    passed = (acc_diff <= 0.005) or (match_fraction >= 0.98)
    status = "pass" if passed else "fail"
    
    print(f"Verification Status: {status.upper()}")
    
    # Read SHA256 checksum if available
    sha256 = "unknown"
    try:
        with open('reports/model_checksum.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if 'SHA256' in line and 'Digit_Recognizer.keras' in line:
                    sha256 = line.split()[0]
                    break
    except:
        pass

    report = {
        "model_sha256": sha256,
        "keras_accuracy": keras_acc,
        "tfjs_accuracy": tfjs_acc,
        "accuracy_diff": acc_diff,
        "label_match_fraction": match_fraction,
        "status": status,
        "site_url": "tbd"
    }
    
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Saved verification report to {report_json}")
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Model Results")
    parser.add_argument('--keras', type=str, default='scripts/keras_results.json')
    parser.add_argument('--tfjs', type=str, default='scripts/tfjs_results.json')
    parser.add_argument('--report', type=str, default='reports/verification_report.json')
    args = parser.parse_args()
    
    main(args.keras, args.tfjs, args.report)
