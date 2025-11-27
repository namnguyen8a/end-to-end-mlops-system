set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Fixing permissions under $REPO_ROOT/data for Airflow uid 50000..."
sudo chown -R 50000:0 "$REPO_ROOT/data"
sudo chmod -R ug+rwX "$REPO_ROOT/data"