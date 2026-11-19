# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from argparse import ArgumentParser, Namespace
from openjd.model import (
    DecodeValidationError,
    JobTemplate,
    TemplateSpecificationVersion,
    decode_job_template,
    decode_environment_template,
)

from .._common import (
    add_extensions_argument,
    read_template,
    OpenJDCliResult,
    print_cli_result,
    process_extensions_argument,
)
from .._common._approvals import is_approval_only, step_approval, terminal_safe


def add_check_arguments(run_parser: ArgumentParser):
    add_extensions_argument(run_parser)


def _approvals_note(job_template: JobTemplate) -> str:
    """Lists the Steps of a Job Template that need approval (STEP_APPROVALS), so that
    whoever checks it can see where a run will wait for a person."""
    lines = [
        f"  - '{terminal_safe(step.name)}'" + (" (approval-only)" if is_approval_only(step) else "")
        for step in job_template.steps
        if step_approval(step) is not None
    ]
    if not lines:
        return ""
    return "\nSteps that need approval before they run:\n" + "\n".join(lines)


@print_cli_result
def do_check(args: Namespace) -> OpenJDCliResult:
    """Open a provided template file and check its schema for errors."""

    extensions = process_extensions_argument(args.extensions)
    note = ""

    try:
        # Raises: RuntimeError
        template_object = read_template(args.path)

        # Raises: KeyError
        document_version = template_object["specificationVersion"]

        # Raises: ValueError
        template_version = TemplateSpecificationVersion(document_version)

        # Raises: DecodeValidationError
        if TemplateSpecificationVersion.is_job_template(template_version):
            job_template = decode_job_template(
                template=template_object, supported_extensions=extensions
            )
            note = _approvals_note(job_template)
        elif TemplateSpecificationVersion.is_environment_template(template_version):
            # Pass `extensions` so env templates can declare e.g.
            # WRAP_ACTIONS (RFC 0008) — without this the model rejects
            # the extension declaration even though the CLI accepts it.
            decode_environment_template(template=template_object, supported_extensions=extensions)
        else:
            return OpenJDCliResult(
                status="error",
                message=f"Unknown template 'specificationVersion' ({document_version}).",
            )

    except KeyError:
        return OpenJDCliResult(
            status="error", message="ERROR: Missing field 'specificationVersion'"
        )
    except RuntimeError as exc:
        return OpenJDCliResult(status="error", message=f"ERROR: {str(exc)}")
    except DecodeValidationError as exc:
        return OpenJDCliResult(
            status="error", message=f"ERROR: '{str(args.path)}' failed checks: {str(exc)}"
        )

    return OpenJDCliResult(
        status="success",
        message=f"Template at '{str(args.path)}' passes validation checks." + note,
    )
