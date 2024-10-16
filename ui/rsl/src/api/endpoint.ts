import getQueryParameters from "@/util/queryParameters";

function getApiEndpointFromUrl(params: Record<string, string>) {
  if (typeof window === "undefined") {
    return "";
  }
  const defaultProtocol = window.location.protocol;
  const defaultHost = window.location.hostname;
  const defaultPort = "8080";
  const motisParam = params["motis"] || null;
  let apiEndpoint = String(window.location.origin + window.location.pathname);
  if (motisParam) {
    if (/^[0-9]+$/.test(motisParam)) {
      apiEndpoint = defaultProtocol + "//" + defaultHost + ":" + motisParam;
    } else if (!motisParam.includes(":")) {
      apiEndpoint = defaultProtocol + "//" + motisParam + ":" + defaultPort;
    } else if (
      !motisParam.startsWith("http:") &&
      !motisParam.startsWith("https:")
    ) {
      apiEndpoint = defaultProtocol + "//" + motisParam;
    } else {
      apiEndpoint = motisParam;
    }
  }

  if (!apiEndpoint.endsWith("/")) {
    apiEndpoint += "/";
  }
  return apiEndpoint;
}

let apiEndpoint = getApiEndpointFromUrl(getQueryParameters());

export function getApiEndpoint(): string {
  return apiEndpoint;
}

export function setApiEndpoint(url: string) {
  apiEndpoint = url;
}
