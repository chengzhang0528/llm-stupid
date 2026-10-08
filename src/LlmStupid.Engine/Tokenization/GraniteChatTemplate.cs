namespace LlmStupid.Engine.Tokenization;

public sealed record ChatMessage(string Role, string Content);

public static class GraniteChatTemplate
{
    private const string DefaultSystemPrompt = "Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. You are a helpful AI assistant.";

    public static string Render(IReadOnlyList<ChatMessage> messages, string? systemPrompt = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0) throw new ArgumentException("At least one chat message is required.", nameof(messages));

        var firstIsSystem = messages[0].Role.Equals("system", StringComparison.OrdinalIgnoreCase);
        var actualSystem = systemPrompt ?? (firstIsSystem ? messages[0].Content : DefaultSystemPrompt);
        var firstConversationMessage = firstIsSystem && systemPrompt is null ? 1 : 0;
        var output = new System.Text.StringBuilder();
        output.Append("<|start_of_role|>system<|end_of_role|>").Append(actualSystem).Append("<|end_of_text|> ");

        for (var i = firstConversationMessage; i < messages.Count; i++)
        {
            var message = messages[i];
            if (message.Role.Equals("system", StringComparison.OrdinalIgnoreCase)) continue;
            output.Append("<|start_of_role|>").Append(message.Role).Append("<|end_of_role|>")
                .Append(message.Content).Append("<|end_of_text|> ");
        }

        output.Append("<|start_of_role|>assistant<|end_of_role|>");
        return output.ToString();
    }
}
