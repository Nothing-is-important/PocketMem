package com.pocketrag.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pocketrag.network.ApiClient
import com.pocketrag.network.AskRequest
import com.pocketrag.network.AskResponse
import kotlinx.coroutines.launch

// ── 消息数据类 ──

data class ChatMessage(
    val content: String,
    val isUser: Boolean,
    val metadata: String = ""
)

// ── 主聊天界面 ──

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen() {
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()

    var messages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var query by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("就绪") }

    // 启动时检查后端连接
    LaunchedEffect(Unit) {
        try {
            val health = ApiClient.getApi().health()
            statusText = "后端已连接 | 状态: ${health["status"]}"
        } catch (e: Exception) {
            statusText = "连接失败: ${e.message}"
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // 标题栏
        TopAppBar(
            title = { Text("PocketMemory") },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
        )

        // 状态栏
        Text(
            text = statusText,
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )

        // 消息列表
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages) { msg ->
                MessageBubble(message = msg)
            }

            if (isLoading) {
                item {
                    Text(
                        text = "思考中...",
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(16.dp)
                    )
                }
            }
        }

        // 输入栏
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("搜索你的记忆...") },
                modifier = Modifier.weight(1f),
                enabled = !isLoading,
                singleLine = true
            )

            Spacer(modifier = Modifier.width(8.dp))

            Button(
                onClick = {
                    if (query.isNotBlank() && !isLoading) {
                        val q = query.trim()
                        messages = messages + ChatMessage(content = q, isUser = true)
                        query = ""
                        isLoading = true
                        statusText = "搜索中..."

                        scope.launch {
                            try {
                                val response = ApiClient.getApi().ask(AskRequest(q))
                                val meta = "意图: ${response.intent} | 延迟: ${response.latency_ms.toInt()}ms"
                                messages = messages + ChatMessage(
                                    content = response.answer,
                                    isUser = false,
                                    metadata = meta
                                )
                                statusText = "就绪 | 延迟: ${response.latency_ms.toInt()}ms"
                            } catch (e: Exception) {
                                messages = messages + ChatMessage(
                                    content = "查询失败: ${e.message}",
                                    isUser = false
                                )
                                statusText = "查询失败: ${e.message}"
                            } finally {
                                isLoading = false
                            }
                        }
                    }
                },
                enabled = !isLoading && query.isNotBlank()
            ) {
                Text("发送")
            }
        }
    }
}

// ── 消息气泡 ──

@Composable
fun MessageBubble(message: ChatMessage) {
    val bgColor = if (message.isUser)
        MaterialTheme.colorScheme.primaryContainer
    else
        MaterialTheme.colorScheme.secondaryContainer

    val alignment = if (message.isUser)
        Alignment.End
    else
        Alignment.Start

    Column(
        horizontalAlignment = alignment,
        modifier = Modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(12.dp))
                .background(bgColor)
                .padding(12.dp)
        ) {
            Column {
                Text(
                    text = message.content,
                    fontSize = 15.sp
                )
                if (message.metadata.isNotEmpty()) {
                    Text(
                        text = message.metadata,
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
