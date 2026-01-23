import type { DefinitionsRegistry } from "@/types/nexus";

export function definitionTooltip(definitions: DefinitionsRegistry | undefined, key: string) {
    if (!definitions || !definitions[key]) return undefined;
    const def = definitions[key];
    const parts = [def.definition_md];
    if (def.assumptions) {
        parts.push(`Assumptions: ${def.assumptions}`);
    }
    if (def.warnings) {
        parts.push(`Warnings: ${def.warnings}`);
    }
    return parts.join(" ");
}
