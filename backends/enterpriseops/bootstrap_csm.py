"""Pinned CSM exact integer queries and lossless HTTP error propagation."""
import inspect
import json
import os
from functools import wraps

if __package__:
    from .backend_errors import SQLITE_INT_MIN, SQLITE_INT_MAX, patch_mcp_result_capture
else:
    from backend_errors import SQLITE_INT_MIN, SQLITE_INT_MAX, patch_mcp_result_capture


def patch_product_integer_queries(manager_class):
    """Preserve the unbounded integer schema and compare without lossy binding."""
    original = manager_class.find_products
    if tuple(inspect.signature(original).parameters) != ('self', 'db_id', 'filters'):
        raise RuntimeError('pinned CSM product query signature differs')

    @wraps(original)
    def find_products(self, db_id, **filters):
        oversized = {name for name in ('product_id', 'product_price')
                     if isinstance(filters.get(name), int)
                     and not SQLITE_INT_MIN <= filters[name] <= SQLITE_INT_MAX}
        if not oversized:
            return original(self, db_id, **filters)
        connection = self._connect(db_id)
        try:
            # The decimal argument travels losslessly as text. Python's numeric
            # equality compares int/int and int/float exactly, without rounding
            # the requested integer. Actual DB errors still propagate.
            connection.create_function('robustops_integer_equal', 2,
                                       lambda stored, decimal: int(stored == int(decimal)),
                                       deterministic=True)
            columns = {'product_id': 'product_id', 'name': 'name',
                       'category': 'category', 'lifecycle_state': 'lifecycle_state',
                       'product_price': 'product_price', 'created_before': 'sys_created_on',
                       'created_after': 'sys_created_on'}
            clauses, parameters = [], []
            for name, column in columns.items():
                value = filters.get(name)
                if value is None:
                    continue
                if name in oversized:
                    clauses.append(f'robustops_integer_equal({column}, ?)')
                    parameters.append(str(value))
                else:
                    operator = {'name': 'LIKE', 'created_before': '<=', 'created_after': '>='}.get(name, '=')
                    clauses.append(f'{column} {operator} ?')
                    parameters.append(f'%{value}%' if name == 'name' else value)
            rows = connection.execute(
                'SELECT product_id, name, category, lifecycle_state, product_price, sys_created_on '
                'FROM product WHERE ' + ' AND '.join(clauses) + ' ORDER BY product_id', parameters).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()
    manager_class.find_products = find_products


def patch_http_errors(module):
    """Keep the native request and success body; retain failures before flattening."""
    original = module._post_tool
    if tuple(inspect.signature(original).parameters) != ('tool', 'args', 'database_id'):
        raise RuntimeError('pinned CSM HTTP tool signature differs')

    @wraps(original)
    async def post_tool(tool, args, database_id):
        payload = {'database_id': database_id, **(args or {})}
        try:
            async with module.httpx.AsyncClient() as client:
                response = await client.post(f'{module.API_BASE_URL}/tools/{tool}', json=payload)
            response.raise_for_status()
            return {'text': json.dumps(response.json()), 'isError': False}
        except module.httpx.HTTPStatusError as exc:
            response = exc.response
            try:
                body = response.json()
                detail = body.get('detail', body) if isinstance(body, dict) else body
            except ValueError:
                detail = response.text
            return {
                'text': f'API Error [{tool}]: {detail}', 'isError': True,
                'status_code': response.status_code,
                'error': {'code': response.status_code, 'message': str(detail), 'details': detail},
            }
        except module.httpx.RequestError as exc:
            return {'text': f'Request Error [{tool}]: {exc}', 'isError': True,
                    'status_code': 502, 'error': {'code': 'BACKEND_TRANSPORT_ERROR', 'message': str(exc)}}
        # Unexpected conversion/programming failures must escape to MCP's
        # infrastructure boundary, not become an ordinary model tool failure.
        except Exception as exc:
            return {'text': f'Tool execution failed [{tool}]: {exc}', 'isError': True,
                    'status_code': 500, 'error': {'code': 'BACKEND_IMPLEMENTATION_ERROR', 'message': str(exc)}}
    module._post_tool = post_tool


def main():
    from database.csm_manager import CSMManager
    patch_product_integer_queries(CSMManager)
    from handlers import tool_handlers
    patch_http_errors(tool_handlers)
    from handlers import mcp_handler
    patch_mcp_result_capture(mcp_handler)
    import main as backend
    import uvicorn
    uvicorn.run(backend.app, host='0.0.0.0', port=int(os.environ.get('API_PORT', '8005')))


if __name__ == '__main__':
    main()
