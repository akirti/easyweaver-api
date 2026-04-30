/**
 * Validate query execution request payload (POST /queries/execute).
 *
 * Ensures the request body contains required fields for a valid query:
 *   - type: query type (e.g., "simple", "join")
 *   - left: object with source_id and table
 *
 * For join queries, also validates:
 *   - right: object with source_id and table
 *   - join_config: object with join type and conditions
 */

var contentType = context.getVariable("request.header.Content-Type");
var requestBody = context.getVariable("request.content");

// Check content type
if (!contentType || contentType.indexOf("application/json") === -1) {
    context.setVariable("validation.error", true);
    context.setVariable("validation.errorMessage", "Content-Type must be application/json");
    context.setVariable("validation.errorCode", "INVALID_CONTENT_TYPE");
    throw new Error("Invalid Content-Type");
}

try {
    var payload = JSON.parse(requestBody);
    var errors = [];

    // Validate type
    if (!payload.type) {
        errors.push("Query type is required");
    } else {
        var validTypes = ["simple", "join", "multi_join"];
        var typeFound = false;
        for (var i = 0; i < validTypes.length; i++) {
            if (validTypes[i] === payload.type) {
                typeFound = true;
                break;
            }
        }
        if (!typeFound) {
            errors.push("Query type must be one of: simple, join, multi_join");
        }
    }

    // Validate left (primary query source)
    if (!payload.left) {
        errors.push("Left (primary) query source is required");
    } else {
        if (!payload.left.source_id) {
            errors.push("Left source_id is required");
        }
        if (!payload.left.table) {
            errors.push("Left table is required");
        }
    }

    // For join queries, validate right and join_config
    if (payload.type === "join" || payload.type === "multi_join") {
        if (!payload.right) {
            errors.push("Right query source is required for join queries");
        } else {
            if (!payload.right.source_id) {
                errors.push("Right source_id is required");
            }
            if (!payload.right.table) {
                errors.push("Right table is required");
            }
        }

        if (!payload.join_config) {
            errors.push("Join configuration is required for join queries");
        }
    }

    // Validate max_rows if provided (must be positive integer)
    if (payload.max_rows !== undefined && payload.max_rows !== null) {
        var maxRows = parseInt(payload.max_rows);
        if (isNaN(maxRows) || maxRows <= 0) {
            errors.push("max_rows must be a positive integer");
        } else if (maxRows > 1000000) {
            errors.push("max_rows cannot exceed 1,000,000");
        }
    }

    // Check for validation errors
    if (errors.length > 0) {
        context.setVariable("validation.error", true);
        context.setVariable("validation.errorMessage", errors.join(", "));
        context.setVariable("validation.errorCode", "VALIDATION_ERROR");
        throw new Error(errors.join(", "));
    }

    // Set validated data for logging
    context.setVariable("query.type", payload.type);
    if (payload.left && payload.left.source_id) {
        context.setVariable("query.source_id", payload.left.source_id);
    }

} catch (e) {
    if (!context.getVariable("validation.error")) {
        context.setVariable("validation.error", true);
        context.setVariable("validation.errorMessage", "Invalid JSON payload");
        context.setVariable("validation.errorCode", "INVALID_JSON");
    }
    throw e;
}
