import importlib
from typing import Any

from dsl.models import AssertRule, RunConfig, Scenario
from runtime.operators import OPERATORS, OpResult
from runtime.query import eval_path


class Engine:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def run_inference(self, run_config: RunConfig, input_data: dict[str, Any]) -> dict[str, Any]:
        """Выполняет инференс согласно run конфигурации"""
        if run_config.kind == "python":
            # Импортируем функцию по пути module.function
            module_path, func_name = run_config.target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func(input_data)
        else:
            raise NotImplementedError(f"Run kind '{run_config.kind}' not implemented")

    def eval_assert(self, rule: AssertRule, document: dict[str, Any]) -> tuple[bool, str]:
        """
        Выполняет одно утверждение.
        Возвращает (ok, message)
        """

        if rule.all_:
            results = [self.eval_assert(r, document) for r in rule.all_]
            all_ok = all(ok for ok, _ in results)
            messages = [msg for ok, msg in results if not ok]
            return all_ok, "; ".join(messages) if messages else "All checks passed"

        if rule.any_:
            results = [self.eval_assert(r, document) for r in rule.any_]
            any_ok = any(ok for ok, _ in results)
            if any_ok:
                return True, "At least one check passed"
            messages = [msg for _, msg in results]
            return False, f"None passed: {'; '.join(messages)}"

        if rule.not_:
            ok, msg = self.eval_assert(rule.not_, document)
            return not ok, f"NOT failed: {msg}" if ok else "NOT passed"

        if not rule.path:
            return False, "No path specified"

        selection = eval_path(document, rule.path)

        op_func = OPERATORS.get(rule.op)
        if not op_func:
            return False, f"Unknown operator: {rule.op}"

        params = {"expected": rule.expected}
        result: OpResult = op_func(selection, params)

        return result.ok, result.message or "OK"

    def run_test(self, test_case: Scenario) -> dict[str, Any]:
        """Выполняет один тест-кейс"""
        case_id = test_case.case.get("id", "unknown")
        print(f"\nRunning: {case_id}")

        try:
            # Шаг 1: Запускаем инференс
            result = self.run_inference(test_case.run, test_case.input)
            print("Inference completed")

            # Шаг 2: Выполняем все проверки
            assert_results = []
            for i, assert_rule in enumerate(test_case.asserts):
                ok, message = self.eval_assert(assert_rule, result)
                assert_results.append(
                    {
                        "index": i,
                        "ok": ok,
                        "message": message,
                        "rule": assert_rule.model_dump(),
                    }
                )

                status = "✓" if ok else "✗"
                print(f"  {status} Assert {i}: {message}")

            # Шаг 3: Определяем итоговый статус
            all_passed = all(r["ok"] for r in assert_results)

            return {
                "case_id": case_id,
                "status": "PASS" if all_passed else "FAIL",
                "asserts": assert_results,
                "result": result,
            }

        except Exception as e:
            print(f"ERROR: {e}")
            return {"case_id": case_id, "status": "ERROR", "error": str(e)}
