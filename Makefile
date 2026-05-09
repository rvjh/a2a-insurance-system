.PHONY: install run-claims run-policy run-gateway run test fmt lint clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run-claims:
	uvicorn agents.claims_agent.main:app --reload --port 8001

run-policy:
	uvicorn agents.policy_agent.main:app --reload --port 8002

run-gateway:
	uvicorn gateway.main:app --reload --port 8000

# Run all 3 in background (Linux/macOS). Use `make stop` to kill.
run:
	@echo "Starting claims agent on :8001"
	@uvicorn agents.claims_agent.main:app --port 8001 > /tmp/claims.log 2>&1 &
	@echo "Starting policy agent on :8002"
	@uvicorn agents.policy_agent.main:app --port 8002 > /tmp/policy.log 2>&1 &
	@sleep 3
	@echo "Starting gateway on :8000"
	@uvicorn gateway.main:app --port 8000

stop:
	@pkill -f 'uvicorn agents.claims_agent.main' || true
	@pkill -f 'uvicorn agents.policy_agent.main' || true
	@pkill -f 'uvicorn gateway.main' || true

demo:
	python -m tests.client_demo

test:
	pytest tests/ -v

fmt:
	black .
	ruff check --fix .

lint:
	ruff check .
	mypy --ignore-missing-imports common gateway agents mcp_tools

clean:
	rm -rf reports/*.html __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
