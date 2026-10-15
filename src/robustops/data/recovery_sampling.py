"""Teacher guidance, student context counting and collection-pool membership."""
from robustops.prompts import read_prompt


def teacher_guidance(variant='consistent_narration'):
    if variant == 'none':
        return ''
    if variant == 'consistent_narration':
        return read_prompt('teacher.md')
    raise ValueError('unknown teacher guidance variant')


TEACHER_GUIDANCE = teacher_guidance()


class StudentContextCounter:
    """Budget student-visible input; teacher API usage stays in provider logs."""
    def __init__(self, counter, original_system, guidance):
        self.counter = counter
        self.original_system = original_system
        self.teacher_system = original_system + ('\n\n' + guidance if guidance else '')

    def text(self, content):
        return self.counter.text(self.original_system if content == self.teacher_system else content)

    def context(self, messages, tools):
        if not messages or messages[0] != {'role': 'system', 'content': self.teacher_system}:
            raise ValueError('unexpected teacher prompt during student context count')
        visible = [{'role': 'system', 'content': self.original_system}, *messages[1:]]
        return self.counter.context(visible, tools)


def require_sft_member(pool, task_id, mode):
    matches = [r for r in pool['tasks'] if r['task_id'] == task_id]
    if len(matches) != 1 or matches[0]['pool'] != 'sft' or mode not in matches[0]['modes']:
        raise ValueError(f'task/mode outside frozen Recovery SFT pool: {task_id}/{mode}')


def teacher_guidance_fragments(guidance):
    return [block for block in guidance.split('\n\n') if block.strip()]
