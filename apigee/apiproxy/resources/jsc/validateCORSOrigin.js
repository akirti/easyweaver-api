/**
 * Validate the request Origin header against an allowed list.
 *
 * Allowed origins are read from the KVM variable "kvm.cors.allowed_origins"
 * (comma-separated) with a sensible default fallback.
 *
 * Sets flow variables:
 *   "cors.allowed_origin" — the validated origin (or empty if not allowed)
 *   "cors.is_valid_origin" — "true" / "false" for use in Conditions
 */

const origin = context.getVariable("request.header.origin") || "";
const allowedOriginsRaw = context.getVariable("kvm.cors.allowed_origins") || "";

// Default allowed origins when KVM entry is not configured.
const DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8003",
    "http://localhost",
    "http://localhost:80",
    "https://akirtidev.mydomain.com",
    "https://easylifeqa.localhost.net",
    "https://easylifestg.localhost.net"
];

// Wildcard suffixes: any origin whose hostname ends with these is allowed.
const ALLOWED_SUFFIXES = [
    ".mydomain.com",
    ".localhost.net"
];

let allowedOrigins;
let usingKVM = false;
if (allowedOriginsRaw && allowedOriginsRaw.trim().length > 0) {
    allowedOrigins = allowedOriginsRaw.split(",").map(function (o) {
        return o.trim();
    });
    usingKVM = true;
} else {
    allowedOrigins = DEFAULT_ORIGINS;
}

let validatedOrigin = "";
let isValid = false;

if (origin) {
    // 1. Exact match
    for (let i = 0; i < allowedOrigins.length; i++) {
        if (allowedOrigins[i] === origin) {
            validatedOrigin = origin;
            isValid = true;
            break;
        }
    }

    // 2. Suffix match (wildcard subdomains)
    if (!isValid) {
        const originWithoutProto = origin.replace(/^https?:\/\//, "");
        const hostname = originWithoutProto.split(":")[0];
        for (let j = 0; j < ALLOWED_SUFFIXES.length; j++) {
            const suffix = ALLOWED_SUFFIXES[j];
            if (hostname.length > suffix.length &&
                hostname.endsWith(suffix)) {
                validatedOrigin = origin;
                isValid = true;
                break;
            }
        }
    }

    // 3. Fallback: if KVM is NOT configured and origin is HTTPS,
    //    allow it (production may have origins not yet in the default list).
    //    When KVM IS configured, strictly enforce the list.
    if (!isValid && !usingKVM && origin.startsWith("https://")) {
        validatedOrigin = origin;
        isValid = true;
        // Log so it's visible in Apigee analytics
        context.setVariable("cors.fallback_origin", origin);
    }
} else {
    // No Origin header (server-to-server, health probes, curl) — allow through
    // without CORS headers (non-browser requests don't need CORS)
    isValid = true;
}

context.setVariable("cors.allowed_origin", validatedOrigin);
context.setVariable("cors.is_valid_origin", isValid ? "true" : "false");
