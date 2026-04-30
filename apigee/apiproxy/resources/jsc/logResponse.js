/**
 * Log response data for analytics
 * Captures response time, status, and key metrics
 */

// Calculate response time
var startTime = context.getVariable("client.received.start.timestamp");
var endTime = context.getVariable("system.timestamp");
var responseTime = endTime - startTime;

// Get response details
var statusCode = context.getVariable("response.status.code");
var targetStatusCode = context.getVariable("target.response.status.code");
var messageId = context.getVariable("messageid");

// Set analytics variables
context.setVariable("analytics.responseTime", responseTime);
context.setVariable("analytics.statusCode", statusCode);
context.setVariable("analytics.targetStatusCode", targetStatusCode);
context.setVariable("analytics.messageId", messageId);

// Determine response category
var category;
if (statusCode >= 200 && statusCode < 300) {
    category = "success";
} else if (statusCode >= 400 && statusCode < 500) {
    category = "client_error";
} else if (statusCode >= 500) {
    category = "server_error";
} else {
    category = "other";
}
context.setVariable("analytics.responseCategory", category);

// Log key details for debugging (in Apigee trace)
print("Request ID: " + messageId);
print("Response Time: " + responseTime + "ms");
print("Status Code: " + statusCode);
print("Category: " + category);

// Set custom dimensions for analytics
context.setVariable("analytics.apiName", context.getVariable("apiproxy.name"));
context.setVariable("analytics.apiRevision", context.getVariable("apiproxy.revision"));
context.setVariable("analytics.environment", context.getVariable("environment.name"));
context.setVariable("analytics.clientIp", context.getVariable("client.ip"));
context.setVariable("analytics.userAgent", context.getVariable("request.header.User-Agent"));
context.setVariable("analytics.verb", context.getVariable("request.verb"));
context.setVariable("analytics.path", context.getVariable("proxy.pathsuffix"));
