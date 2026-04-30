/**
 * Extract access_token from Cookie header and promote it to Authorization header.
 *
 * When the browser sends an httpOnly cookie (access_token=<jwt>) but no
 * Authorization header, this script copies the cookie value into the
 * Authorization header so the downstream JWT-VerifyAccessToken policy
 * can validate it.
 *
 * Runs in PreFlow BEFORE JWT-VerifyAccessToken.
 */
var authHeader = context.getVariable("request.header.Authorization");

// Only act when Authorization header is absent or empty
if (!authHeader) {
    var cookieHeader = context.getVariable("request.header.Cookie");
    if (cookieHeader) {
        // Parse cookies: "access_token=eyJ...; refresh_token=eyJ...; other=val"
        var cookies = cookieHeader.split(";");
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.indexOf("access_token=") === 0) {
                var token = cookie.substring("access_token=".length);
                if (token) {
                    context.setVariable("request.header.Authorization", "Bearer " + token);
                }
                break;
            }
        }
    }
}
